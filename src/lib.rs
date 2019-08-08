// TODO: make sure all this unsafe is actually okay
extern crate gfx_hal as hal;
extern crate imgui;
#[macro_use]
extern crate failure;
#[macro_use]
extern crate memoffset;

use std::borrow::BorrowMut;
use std::iter;
use std::mem;

use hal::memory::Properties;
use hal::pso::{PipelineStage, Rect};
use hal::{
    buffer, command, device, format, image, memory, pass, pso, queue, Backend,
    CommandQueue, DescriptorPool, Device, MemoryTypeId, PhysicalDevice,
    Primitive,
};
use imgui::{ImDrawIdx, ImDrawVert, ImGui, Ui};
use imgui::{DrawCmd, DrawCmdParams, DrawData, ImString, TextureId, Textures};

#[derive(Clone, Debug, Fail)]
pub enum Error {
    #[fail(display = "can't find valid memory type for {}", _0)]
    CantFindMemoryType(&'static str),

    #[fail(display = "buffer creation error")]
    BufferCreationError(#[cause] buffer::CreationError),

    #[fail(display = "image creation error")]
    ImageCreationError(#[cause] image::CreationError),

    #[fail(display = "bind error")]
    BindError(#[cause] device::BindError),

    #[fail(display = "image view error")]
    ImageViewError(#[cause] image::ViewError),

    #[fail(display = "execution error")]
    ExecutionError(#[cause] hal::error::HostExecutionError),

    #[fail(display = "pipeline allocation error")]
    PipelineAllocationError(#[cause] pso::AllocationError),

    #[fail(display = "allocation error")]
    AllocationError(#[cause] hal::device::AllocationError),

    #[fail(display = "out of memory")]
    OutOfMemoryError(#[cause] hal::device::OutOfMemory),

    #[fail(display = "pipeline creation error")]
    PipelineCreationError(#[cause] pso::CreationError),

    #[fail(display = "mapping error")]
    MappingError(#[cause] hal::mapping::Error),

    #[fail(display = "device lost")]
    DeviceLostError(#[cause] hal::device::DeviceLost),

    // ShaderError doesn't implement Fail, for some reason
    #[fail(display = "shader error: {:?}", _0)]
    ShaderError(device::ShaderError),
}

macro_rules! impl_from {
    ($t:ty, $other:ty, $variant:path) => {
        impl From<$other> for $t {
            fn from(err: $other) -> $t {
                $variant(err)
            }
        }
    };
}

impl_from!(Error, image::ViewError, Error::ImageViewError);
impl_from!(Error, buffer::CreationError, Error::BufferCreationError);
impl_from!(Error, image::CreationError, Error::ImageCreationError);
impl_from!(Error, device::BindError, Error::BindError);
impl_from!(Error, hal::error::HostExecutionError, Error::ExecutionError);
impl_from!(Error, pso::AllocationError, Error::PipelineAllocationError);
impl_from!(Error, hal::device::AllocationError, Error::AllocationError);
impl_from!(Error, hal::device::OutOfMemory, Error::OutOfMemoryError);
impl_from!(Error, pso::CreationError, Error::PipelineCreationError);
impl_from!(Error, device::ShaderError, Error::ShaderError);
impl_from!(Error, hal::mapping::Error, Error::MappingError);

// TODO am I actually using failure correctly?
impl From<hal::device::OomOrDeviceLost> for Error {
    fn from(err: hal::device::OomOrDeviceLost) -> Error {
        use hal::device::OomOrDeviceLost::*;
        match err {
            OutOfMemory(err) => Error::OutOfMemoryError(err),
            DeviceLost(err) => Error::DeviceLostError(err),
        }
    }
}

// TODO: using a separate memory allocation for each frame's buffer
// set is not great. The main issue with sharing memory for both frames is that
// the old memory can't be freed until both frames are complete.
pub struct Buffers<B: Backend> {
    memory: B::Memory,
    mapped: *mut u8,
    vertex_buffer: B::Buffer,
    index_buffer: B::Buffer,
    index_offset: u64,
    num_verts: usize,
    num_inds: usize,
}

pub struct Renderer<B: Backend> {
    sampler: B::Sampler,
    image_memory: B::Memory,
    memory_type_buffers: Option<MemoryTypeId>,
    buffers: Vec<Option<Buffers<B>>>,
    image: B::Image,
    image_view: B::ImageView,
    descriptor_pool: B::DescriptorPool,
    descriptor_set_layout: B::DescriptorSetLayout,
   // descriptor_set: B::DescriptorSet,
    pipeline: B::GraphicsPipeline,
    pipeline_layout: B::PipelineLayout,
    texture_sets : Vec<B::DescriptorSet>
}

impl<B: Backend> Buffers<B> {
    /// Allocates a new pair of vertex and index buffers
    fn new(
        memory_type: &mut Option<MemoryTypeId>,
        num_verts: usize,
        num_inds: usize,
        device: &B::Device,
        physical_device: &B::PhysicalDevice,
    ) -> Result<Buffers<B>, Error> {
        unsafe {
            // Calculate required size for each buffer. Note that total
            // size cannot be calculated because of alignment between the
            // buffers.
            let verts_size = (num_verts * mem::size_of::<ImDrawVert>()) as u64;
            let inds_size = (num_inds * mem::size_of::<ImDrawIdx>()) as u64;

            // Create buffers.
            let mut vertex_buffer =
                device.create_buffer(verts_size, buffer::Usage::VERTEX)?;
            let mut index_buffer =
                device.create_buffer(inds_size, buffer::Usage::INDEX)?;
            let vertex_requirements =
                device.get_buffer_requirements(&vertex_buffer);
            let index_requirements =
                device.get_buffer_requirements(&index_buffer);
            // The GPU size requirements may be larger than a simple
            // packed buffer.
            let verts_size = vertex_requirements.size;
            let inds_size = index_requirements.size;

            // Determine offset given alignment.
            let index_align = index_requirements.alignment;
            let index_offset = if verts_size % index_align == 0 {
                verts_size
            } else {
                verts_size + index_align - verts_size % index_align
            };
            let size = index_offset + inds_size;

            // Find an applicable memory type.
            let type_mask =
                vertex_requirements.type_mask & index_requirements.type_mask;
            let supported = |id| type_mask & (1u64 << id) != 0;
            let memory_type = match memory_type {
                // The old memory type is still valid.
                &mut Some(MemoryTypeId(id)) if supported(id) => id,
                // There was either no cached type or it was no longer
                // valid for the new buffers.
                memory_type => {
                    let memory_types =
                        physical_device.memory_properties().memory_types;

                    let (ty, _) = memory_types
                        .iter()
                        .enumerate()
                        .find(|(id, mem)| {
                            supported(*id)
                                && mem
                                    .properties
                                    .contains(Properties::CPU_VISIBLE)
                        })
                        .ok_or(Error::CantFindMemoryType(
                            "vertex/index buffers",
                        ))?;
                    *memory_type = Some(MemoryTypeId(ty));
                    ty
                }
            };
            // Allocate and bind memory
            let memory =
                device.allocate_memory(MemoryTypeId(memory_type), size)?;
            device.bind_buffer_memory(&memory, 0, &mut vertex_buffer)?;
            device.bind_buffer_memory(
                &memory,
                index_offset,
                &mut index_buffer,
            )?;

            let mapped = device.map_memory(&memory, 0..size)?;

            Ok(Buffers {
                memory,
                mapped,
                vertex_buffer,
                index_buffer,
                index_offset,
                num_verts,
                num_inds,
            })
        }
    }

    /// Checks if there is room in the buffer to store a number of vertices and
    /// indices without re-allocating.
    fn has_room(&self, num_verts: usize, num_inds: usize) -> bool {
        self.num_verts >= num_verts && self.num_inds >= num_inds
    }

    /// Copies vertex and index data into the buffers.
    fn update(
        &mut self,
        verts: &[ImDrawVert],
        inds: &[ImDrawIdx],
        vertex_offset: usize,
        index_offset: usize,
    ) {
        assert!(self.num_verts >= verts.len() + vertex_offset);
        assert!(self.num_inds >= inds.len() + index_offset);
        unsafe {
            // Copy vertex data.
            let dest = self.mapped.offset(
                (vertex_offset * mem::size_of::<ImDrawVert>()) as isize,
            );
            let src = &verts[0];
            std::ptr::copy_nonoverlapping(
                src,
                dest as *mut ImDrawVert,
                verts.len(),
            );

            // Copy index data.
            let dest = self.mapped.offset(
                (self.index_offset as usize
                    + index_offset * mem::size_of::<ImDrawIdx>())
                    as isize,
            );
            let src = &inds[0];
            std::ptr::copy_nonoverlapping(
                src,
                dest as *mut ImDrawIdx,
                inds.len(),
            );
        }
    }

    /// Destroys the buffer
    fn destroy(self, device: &B::Device) {
        unsafe {
            device.unmap_memory(&self.memory);
            device.destroy_buffer(self.vertex_buffer);
            device.destroy_buffer(self.index_buffer);
            device.free_memory(self.memory);
        }
    }

    /// Flush memory changes to syncrhonize.
    fn flush(&self, device: &B::Device) -> Result<(), Error> {
        unsafe {
            device.flush_mapped_memory_ranges(&[(&self.memory, ..)])?;
        }

        Ok(())
    }
}

impl<B: Backend> Renderer<B> {
  

    pub fn add_texture(&mut self,device : &B::Device,sampler : &B::Sampler, image_view : &B::ImageView, image_layout : &hal::image::Layout) -> TextureId {
            
            let descriptor_set = unsafe { 
                self.descriptor_pool.allocate_set(&self.descriptor_set_layout).expect("could not allocate ds set in add_texture")
                };

            {
                let write = pso::DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 0,
                    array_offset: 0,
                    descriptors: &[pso::Descriptor::CombinedImageSampler(
                        image_view,
                        image::Layout::ShaderReadOnlyOptimal,
                        sampler,
                    )],
                };
                unsafe {
                device.write_descriptor_sets(Some(write));
                }
            }
             
             self.texture_sets.push(descriptor_set);
             let index = self.texture_sets.len() - 1;
        
        
        
        
        
        unsafe {
            std::mem::transmute::<usize,TextureId>(index)
        }
    }
    
    
    pub fn new<C>(
        //imgui: &mut ImGui,
        ctx : &mut imgui::Context,
        device: &B::Device,
        physical_device: &B::PhysicalDevice,
        render_pass: &B::RenderPass,
        subpass_index: usize,
        max_frames: usize,
        command_pool: &mut hal::CommandPool<B, C>,
        queue: &mut CommandQueue<B, C>,
    ) -> Result<Renderer<B>, Error>
    where
        C: queue::Capability + queue::Supports<queue::Transfer>,
    {
        // Fence and command buffer for all transfer operations
        let mut transfer_cbuf =
            command_pool.acquire_command_buffer::<command::OneShot>();
        let transfer_fence = device.create_fence(false)?;

        // Determine memory types to use
        let memory_types = physical_device.memory_properties().memory_types;

   
        let mut fonts = ctx.fonts();
        let handle = fonts.build_rgba32_texture();
        // Copy texture
        let (image_memory, image, image_view, staging_memory, staging_buffer) =
            unsafe { 
                let size = u64::from(handle.width * handle.height * 4);

                // Create target image
                let kind = image::Kind::D2(handle.width, handle.height, 1, 1);
                let format = format::Format::Rgba8Unorm;
                let mut image = device.create_image(
                    kind,
                    1,
                    format,
                    image::Tiling::Optimal,
                    image::Usage::SAMPLED | image::Usage::TRANSFER_DST,
                    image::ViewCapabilities::empty(),
                )?;
                let requirements = device.get_image_requirements(&image);
                // Find valid memory type
                let (memory_type, _) = memory_types
                    .iter()
                    .enumerate()
                    .find(|(id, mem)| {
                        let supported =
                            requirements.type_mask & (1u64 << id) != 0;
                        supported
                            && mem.properties.contains(Properties::DEVICE_LOCAL)
                    })
                    .ok_or(Error::CantFindMemoryType("image"))?;
                let image_memory =
                    device.allocate_memory(MemoryTypeId(memory_type), size)?;
                device.bind_image_memory(&image_memory, 0, &mut image)?;

                let subresource_range = image::SubresourceRange {
                    aspects: format::Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                };

                let image_view = device.create_image_view(
                    &image,
                    image::ViewKind::D2,
                    format,
                    format::Swizzle::NO,
                    subresource_range.clone(),
                )?;

                // Create staging buffer
                let mut staging_buffer =
                    device.create_buffer(size, buffer::Usage::TRANSFER_SRC)?;
                let requirements =
                    device.get_buffer_requirements(&staging_buffer);
                let (memory_type, _) = memory_types
                    .iter()
                    .enumerate()
                    .find(|(id, mem)| {
                        let supported =
                            requirements.type_mask & (1u64 << id) != 0;
                        supported
                            && mem.properties.contains(Properties::CPU_VISIBLE)
                    })
                    .ok_or(Error::CantFindMemoryType("image staging buffer"))?;
                let staging_memory =
                    device.allocate_memory(MemoryTypeId(memory_type), size)?;
                device.bind_buffer_memory(
                    &staging_memory,
                    0,
                    &mut staging_buffer,
                )?;

                // Copy data into the mapped staging buffer
                {
                    let mut map = device
                        .acquire_mapping_writer(&staging_memory, 0..size)?;
                    map.clone_from_slice(handle.data);
                    device.release_mapping_writer(map)?;
                }

                {
                    // Build a command buffer to copy data
                    transfer_cbuf.begin();

                    // Copy staging buffer to the image
                    let image_barrier = memory::Barrier::Image {
                        states: (
                            image::Access::empty(),
                            image::Layout::Undefined,
                        )
                            ..(
                                image::Access::TRANSFER_WRITE,
                                image::Layout::TransferDstOptimal,
                            ),
                        target: &image,
                        families: None,
                        range: subresource_range.clone(),
                    };

                    transfer_cbuf.pipeline_barrier(
                        PipelineStage::TOP_OF_PIPE..PipelineStage::TRANSFER,
                        memory::Dependencies::empty(),
                        &[image_barrier],
                    );

                    transfer_cbuf.copy_buffer_to_image(
                        &staging_buffer,
                        &image,
                        image::Layout::TransferDstOptimal,
                        &[command::BufferImageCopy {
                            buffer_offset: 0,
                            buffer_width: handle.width,
                            buffer_height: handle.height,
                            image_layers: image::SubresourceLayers {
                                aspects: format::Aspects::COLOR,
                                level: 0,
                                layers: 0..1,
                            },
                            image_offset: image::Offset { x: 0, y: 0, z: 0 },
                            image_extent: image::Extent {
                                width: handle.width,
                                height: handle.height,
                                depth: 1,
                            },
                        }],
                    );

                    let image_barrier = memory::Barrier::Image {
                        states: (
                            image::Access::TRANSFER_WRITE,
                            image::Layout::TransferDstOptimal,
                        )
                            ..(
                                image::Access::SHADER_READ,
                                image::Layout::ShaderReadOnlyOptimal,
                            ),
                        target: &image,
                        // TODO: this probably should transfer to the
                        // graphics queue
                        families: None,
                        range: subresource_range.clone(),
                    };
                    transfer_cbuf.pipeline_barrier(
                        PipelineStage::TRANSFER..PipelineStage::FRAGMENT_SHADER,
                        memory::Dependencies::empty(),
                        &[image_barrier],
                    );

                    transfer_cbuf.finish();

                    // Submit to the queue
                    queue.submit_nosemaphores(
                        Some(&transfer_cbuf),
                        Some(&transfer_fence),
                    );
                }

                (
                    image_memory,
                    image,
                    image_view,
                    staging_memory,
                    staging_buffer,
                )
            };

        unsafe {
            // Create font sampler
            let sampler = device.create_sampler(image::SamplerInfo::new(
                image::Filter::Linear,
                image::WrapMode::Clamp,
            ))?;

            // Create descriptor set
            let descriptor_set_layout = device.create_descriptor_set_layout(
                &[pso::DescriptorSetLayoutBinding {
                    binding: 0,
                    ty: pso::DescriptorType::CombinedImageSampler,
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::FRAGMENT,
                    immutable_samplers: true,
                }],
                Some(&sampler),
            )?;

            let mut descriptor_pool = device.create_descriptor_pool(
                100,
                &[pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::CombinedImageSampler,
                    count: 100,
                }],
                 pso::DescriptorPoolCreateFlags::empty()
            )?;

            let descriptor_set =
                descriptor_pool.allocate_set(&descriptor_set_layout)?;

            {
                let write = pso::DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 0,
                    array_offset: 0,
                    descriptors: &[pso::Descriptor::CombinedImageSampler(
                        &image_view,
                        image::Layout::ShaderReadOnlyOptimal,
                        &sampler,
                    )],
                };
                device.write_descriptor_sets(Some(write));
            }

            // Create pipeline
            let pipeline_layout = device.create_pipeline_layout(
                Some(&descriptor_set_layout),
                // 4 is the magic number because there are two 2d vectors
                &[(pso::ShaderStageFlags::VERTEX, 0..4)],
            )?;

            // Create shaders
            let vs_module = {
                let spirv = include_bytes!("../shaders/ui.vert.spirv");
                device.create_shader_module(&spirv[..])?
            };
            let fs_module = {
                let spirv = include_bytes!("../shaders/ui.frag.spirv");
                device.create_shader_module(&spirv[..])?
            };

            // Create pipeline
            let pipeline = {
                let vs_entry = pso::EntryPoint {
                    entry: "main",
                    module: &vs_module,
                    specialization: pso::Specialization::default(),
                };
                let fs_entry = pso::EntryPoint {
                    entry: "main",
                    module: &fs_module,
                    specialization: pso::Specialization::default(),
                };

                let shader_entries = pso::GraphicsShaderSet {
                    vertex: vs_entry,
                    hull: None,
                    domain: None,
                    geometry: None,
                    fragment: Some(fs_entry),
                };

                // Create render pass
                let subpass = pass::Subpass {
                    index: subpass_index,
                    main_pass: render_pass,
                };

                let mut pipeline_desc = pso::GraphicsPipelineDesc::new(
                    shader_entries,
                    Primitive::TriangleList,
                    pso::Rasterizer {
                        cull_face: pso::Face::NONE,
                        ..pso::Rasterizer::FILL
                    },
                    &pipeline_layout,
                    subpass,
                );

                // Enable blending
                pipeline_desc.blender.targets.push(pso::ColorBlendDesc(
                    pso::ColorMask::ALL,
                    pso::BlendState::ALPHA,
                ));

                // Set up vertex buffer
                pipeline_desc.vertex_buffers.push(pso::VertexBufferDesc {
                    binding: 0,
                    stride: mem::size_of::<ImDrawVert>() as u32,
                    rate: hal::pso::VertexInputRate::Vertex,
                });

                // Set up vertex attributes
                // Position
                pipeline_desc.attributes.push(pso::AttributeDesc {
                    location: 0,
                    binding: 0,
                    element: pso::Element {
                        format: format::Format::Rg32Sfloat,
                        offset: offset_of!(ImDrawVert, pos) as u32,
                    },
                });
                // UV
                pipeline_desc.attributes.push(pso::AttributeDesc {
                    location: 1,
                    binding: 0,
                    element: pso::Element {
                        format: format::Format::Rg32Sfloat,
                        offset: offset_of!(ImDrawVert, uv) as u32,
                    },
                });
                // Color
                pipeline_desc.attributes.push(pso::AttributeDesc {
                    location: 2,
                    binding: 0,
                    element: pso::Element {
                        format: format::Format::Rgba8Unorm,
                        offset: offset_of!(ImDrawVert, col) as u32,
                    },
                });

                // Create pipeline
                device.create_graphics_pipeline(&pipeline_desc, None)?
            };

            // Clean up shaders
            device.destroy_shader_module(vs_module);
            device.destroy_shader_module(fs_module);

            // Wait until all transfers have finished
            device.wait_for_fence(&transfer_fence, !0)?;

            // Destroy any temporary resources
            command_pool.free(Some(transfer_cbuf));
            device.destroy_fence(transfer_fence);
            device.destroy_buffer(staging_buffer);
            device.free_memory(staging_memory);

            Ok(Renderer {
                sampler,
                memory_type_buffers: None,
                buffers: (0..max_frames).map(|_| None).collect(),
                image_memory,
                image,
                image_view,
                descriptor_pool,
                descriptor_set_layout,
                pipeline,
                pipeline_layout, texture_sets : vec![descriptor_set]
            })
        }
    }

    fn draw<C: BorrowMut<B::CommandBuffer>>(
        &mut self,
        draw_data: &DrawData,
        frame: usize,
        pass: &mut command::RenderSubpassCommon<B, C>,
        device: &B::Device,
        physical_device: &B::PhysicalDevice,
    ) -> Result<(), Error> {
        // Possibly reallocate buffers
        if self.buffers[frame]
            .as_ref()
            .map(|buffers| {
                !buffers.has_room(
                    draw_data.total_vtx_count as usize,
                    draw_data.total_idx_count as usize,
                )
            })
            .unwrap_or(true)
        {
            let num_verts = draw_data.total_vtx_count as usize;
            let num_inds = draw_data.total_idx_count as usize;
            if num_verts == 0 || num_inds == 0 {
                // don't need to do anything
                return Ok(());
            }
            let buffers = Buffers::new(
                &mut self.memory_type_buffers,
                num_verts,
                num_inds,
                device,
                physical_device,
            )?;
            if let Some(old) =
                mem::replace(&mut self.buffers[frame], Some(buffers))
            {
                old.destroy(device);
            }
        }
        let buffers = self.buffers[frame].as_mut().unwrap();
        let mut vertex_offset = 0;
        let mut index_offset = 0;

        unsafe {
            // Bind pipeline
            pass.bind_graphics_pipeline(&self.pipeline);
       

            // Bind vertex and index buffers
            pass.bind_vertex_buffers(
                0,
                iter::once((&buffers.vertex_buffer, 0)),
            );
            pass.bind_index_buffer(buffer::IndexBufferView {
                buffer: &buffers.index_buffer,
                offset: 0,
                index_type: hal::IndexType::U16,
            });

            let (width, height) = (draw_data.display_size[0], draw_data.display_size[1]);

            // Set up viewport
            let viewport = pso::Viewport {
                rect: pso::Rect {
                    x: 0,
                    y: 0,
                    w: width as i16,
                    h: height as i16,
                },
                depth: 0.0..1.0,
            };
            pass.set_viewports(0, &[viewport]);

            // Set push constants
            let push_constants = [
                // scale
                2.0 / width,
                2.0 / height,
                //offset
                -1.0,
                -1.0,
            ];
            // yikes
            let push_constants: [u32; 4] = std::mem::transmute(push_constants);
            pass.push_graphics_constants(
                &self.pipeline_layout,
                pso::ShaderStageFlags::VERTEX,
                0,
                &push_constants,
            );

            // Iterate over drawlists
            for list in draw_data.draw_lists() {
                // Update vertex and index buffers
                buffers.update(
                    list.vtx_buffer(),
                    list.idx_buffer(),
                    vertex_offset,
                    index_offset,
                );

                for cmd in list.commands() {
                        match cmd {
                    DrawCmd::Elements {
                        count,
                        cmd_params:
                            DrawCmdParams {
                                clip_rect,
                                texture_id,
                                ..
                            },
                    } => {
                    //Calculate the scissor
                    let scissor = Rect {
                        x: clip_rect[0] as i16,
                        y: clip_rect[1] as i16,
                        w: (clip_rect[2] - clip_rect[0]) as i16,
                        h: (clip_rect[3] - clip_rect[1]) as i16,
                    };
                    pass.set_scissors(0, &[scissor]);

                    
                 
                
                  let texture_index = texture_id.id();
                  
                  
                     pass.bind_graphics_descriptor_sets(
                &self.pipeline_layout,
                0,
                Some(&self.texture_sets[texture_index]),
                None as Option<u32>,
            );
                

                    // Actually draw things
                    pass.draw_indexed(
                        index_offset as u32
                            ..index_offset as u32 + count as u32,
                        vertex_offset as i32,
                        0..1,
                    );

                    index_offset += count as usize;
                    }
                    DrawCmd::ResetRenderState => (), // TODO
                    DrawCmd::RawCallback { callback, raw_cmd } => unsafe {
                        //(callback(draw_list.raw(), raw_cmd)
                    },
                }
                
                }

                // Increment offsets
                vertex_offset += list.vtx_buffer().len();
            }

            buffers.flush(device)?;

            Ok(())
        }
    }

    /// Renders a frame.
    pub fn render<C: BorrowMut<B::CommandBuffer>>(
        &mut self,
        ui: Ui,
        frame: usize,
        render_pass: &mut command::RenderSubpassCommon<B, C>,
        device: &B::Device,
        physical_device: &B::PhysicalDevice,
    ) -> Result<(), Error> {
       
       let draw_data = ui.render();
       
     
            self.draw(
                &draw_data,
                frame,
                render_pass,
                device,
                physical_device,
            );
    
        Ok(())
    }

    /// Destroys all used objects.
    pub fn destroy(mut self, device: &B::Device) {
        unsafe {
            device.destroy_image(self.image);
            device.free_memory(self.image_memory);
            device.destroy_image_view(self.image_view);
            device.destroy_sampler(self.sampler);
            self.descriptor_pool.reset();
            device.destroy_descriptor_pool(self.descriptor_pool);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout);
            device.destroy_graphics_pipeline(self.pipeline);
            device.destroy_pipeline_layout(self.pipeline_layout);
            for buffers in self.buffers.into_iter() {
                buffers.map(|buffers| buffers.destroy(device));
            }
        }
    }
}
