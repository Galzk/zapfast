//! Releases egui's per-URI image caches for images that leave the screen.
//!
//! egui caches the bytes, the decoded pixels, and the GPU texture of every URI
//! it has loaded. It only ever evicts the textures of `.svg` files, so left
//! alone a session that scrolls through photos holds every one of them for
//! good. Each draw site registers its URI here, and [`sweep`] forgets the ones
//! that fall out of the resident window. The window is bounded by both URI
//! count and offscreen texture bytes: a photo can cost more than many avatars.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Image URIs kept resident. Anything the current frame drew is always kept,
/// so this only bounds what has scrolled away.
const RESIDENT: usize = 96;
const OFFSCREEN_TEXTURE_BYTES: usize = 24 * 1024 * 1024;

#[derive(Default)]
struct TextureCost<'a> {
    bytes: usize,
    /// egui loads GIF and WebP frames as separate texture URIs.
    frames: Vec<&'a str>,
}

fn frame_source(name: &str) -> Option<&str> {
    let (uri, index) = name.rsplit_once('#')?;
    let extension = uri.rsplit_once('.')?.1;
    if extension.eq_ignore_ascii_case("gif") || extension.eq_ignore_ascii_case("webp") {
        index.parse::<usize>().ok().map(|_| uri)
    } else {
        None
    }
}

struct Entry {
    /// Frame the image was last drawn on.
    frame: u64,
    /// Whether its bytes are registered with egui's byte loader. egui drops
    /// them once the texture is uploaded, so this is what says a later draw
    /// has to register them again.
    registered: bool,
}

#[derive(Clone, Default)]
struct Registry(Arc<Mutex<HashMap<String, Entry>>>);

fn registry(ctx: &egui::Context) -> Registry {
    ctx.data_mut(|data| {
        data.get_temp_mut_or_default::<Registry>(egui::Id::new("image-cache"))
            .clone()
    })
}

/// Registers `bytes` under `uri`, and marks the image as drawn this frame.
///
/// Re-registers them after [`sweep`] has forgotten the image, or after egui
/// has dropped them once the texture was uploaded.
pub fn include(ctx: &egui::Context, uri: &str, bytes: &[u8]) {
    let frame = ctx.cumulative_frame_nr();
    let registry = registry(ctx);
    let mut entries = registry.0.lock().unwrap_or_else(|p| p.into_inner());
    let fresh = match entries.get_mut(uri) {
        Some(entry) => {
            entry.frame = frame;
            !std::mem::replace(&mut entry.registered, true)
        }
        None => {
            entries.insert(
                uri.to_owned(),
                Entry {
                    frame,
                    registered: true,
                },
            );
            true
        }
    };
    drop(entries);
    if fresh {
        ctx.include_bytes(uri.to_owned(), bytes.to_vec());
    }
}

/// Marks an image URI as drawn this frame.
pub fn touch(ctx: &egui::Context, uri: &str) {
    let frame = ctx.cumulative_frame_nr();
    let registry = registry(ctx);
    let mut entries = registry.0.lock().unwrap_or_else(|p| p.into_inner());
    match entries.get_mut(uri) {
        Some(entry) => entry.frame = frame,
        None => {
            entries.insert(
                uri.to_owned(),
                Entry {
                    frame,
                    registered: false,
                },
            );
        }
    }
}

/// Forgets one image now rather than once it falls out of the resident
/// window: the info panel frees its thumbnails when it closes. A later draw
/// registers the bytes again.
pub fn forget(ctx: &egui::Context, uri: &str) {
    let registry = registry(ctx);
    registry
        .0
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .remove(uri);
    let frames: Vec<String> = ctx
        .tex_manager()
        .read()
        .allocated()
        .filter(|(_, texture)| frame_source(&texture.name) == Some(uri))
        .map(|(_, texture)| texture.name.clone())
        .collect();
    for frame in frames {
        ctx.forget_image(&frame);
    }
    ctx.forget_image(uri);
}

/// Keeps recent offscreen images within the count and texture byte budgets.
///
/// Images the current frame drew are never forgotten, so a frame that shows
/// more than `RESIDENT` images does not reload them on the next one.
pub fn sweep(ctx: &egui::Context) {
    let frame = ctx.cumulative_frame_nr();
    let stale: Vec<String> = {
        let registry = registry(ctx);
        let mut entries = registry.0.lock().unwrap_or_else(|p| p.into_inner());
        if entries.values().all(|entry| entry.frame == frame) {
            return;
        }

        let manager = ctx.tex_manager();
        let textures = manager.read();
        let mut costs = HashMap::<&str, TextureCost<'_>>::new();
        for (_, texture) in textures.allocated() {
            let name = texture.name.as_str();
            let uri = frame_source(name).unwrap_or(name);
            if entries.get(uri).is_some_and(|entry| entry.frame < frame) {
                let cost = costs.entry(uri).or_default();
                cost.bytes += texture.bytes_used();
                if uri != name {
                    cost.frames.push(name);
                }
            }
        }
        if entries.len() <= RESIDENT
            && costs.values().map(|cost| cost.bytes).sum::<usize>() <= OFFSCREEN_TEXTURE_BYTES
        {
            return;
        }

        let mut idle: Vec<_> = entries
            .iter()
            .filter_map(|(uri, entry)| (entry.frame < frame).then_some((entry.frame, uri)))
            .collect();
        idle.sort_unstable_by_key(|entry| std::cmp::Reverse(*entry));
        let mut slots = RESIDENT.saturating_sub(entries.len() - idle.len());
        let mut bytes = OFFSCREEN_TEXTURE_BYTES;
        let mut stale = Vec::new();
        for (_, uri) in idle {
            let cost = costs.remove(uri.as_str()).unwrap_or_default();
            if slots > 0 && cost.bytes <= bytes {
                slots -= 1;
                bytes -= cost.bytes;
            } else {
                stale.push(uri.clone());
                stale.extend(cost.frames.into_iter().map(str::to_owned));
            }
        }
        for uri in &stale {
            entries.remove(uri);
        }
        stale
    };
    for uri in stale {
        ctx.forget_image(&uri);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> egui::Context {
        let ctx = egui::Context::default();
        egui_extras::install_image_loaders(&ctx);
        ctx
    }

    /// Advances the frame counter the way a repaint would.
    fn frame(ctx: &egui::Context) {
        let mut output = ctx.run_ui(egui::RawInput::default(), |_| {});
        output.textures_delta.clear();
    }

    fn registered(ctx: &egui::Context, uri: &str) -> bool {
        ctx.try_load_bytes(uri).is_ok()
    }

    fn uri(index: usize) -> String {
        format!("bytes://test-{index:04}")
    }

    /// Answers in the same frame, so a test can watch a texture being
    /// allocated without waiting on a decoding thread.
    struct ReadyLoader(Arc<egui::ColorImage>);

    impl egui::load::ImageLoader for ReadyLoader {
        fn id(&self) -> &str {
            "zapfast::image_cache::tests::ReadyLoader"
        }

        fn load(
            &self,
            _ctx: &egui::Context,
            _uri: &str,
            _size_hint: egui::load::SizeHint,
        ) -> egui::load::ImageLoadResult {
            Ok(egui::load::ImagePoll::Ready {
                image: Arc::clone(&self.0),
            })
        }

        fn forget(&self, _uri: &str) {}

        fn forget_all(&self) {}

        fn byte_size(&self) -> usize {
            0
        }
    }

    /// Draws `shown` and then sweeps, the way a frame of the app does.
    fn draw(ctx: &egui::Context, shown: &[String]) -> egui::FullOutput {
        ctx.run_ui(egui::RawInput::default(), |ui| {
            for uri in shown {
                ui.add(egui::Image::from_uri(uri.clone()));
            }
            sweep(ui.ctx());
        })
    }

    fn allocated(ctx: &egui::Context) -> usize {
        let manager = ctx.tex_manager();
        manager.read().num_allocated()
    }

    #[test]
    fn a_forgotten_image_is_released_and_registers_again_when_drawn() {
        let ctx = context();
        include(&ctx, &uri(1), b"image bytes");
        assert!(registered(&ctx, &uri(1)));
        forget(&ctx, &uri(1));
        assert!(
            !registered(&ctx, &uri(1)),
            "forgotten at once, not swept later"
        );
        include(&ctx, &uri(1), b"image bytes");
        assert!(registered(&ctx, &uri(1)), "a later draw registers it again");
    }

    #[test]
    fn images_the_current_frame_drew_are_never_released() {
        let ctx = context();
        let total = RESIDENT + 40;
        for index in 0..total {
            include(&ctx, &uri(index), b"image bytes");
        }
        // A frame that shows more images than the window holds must not drop
        // the ones it is showing right now.
        sweep(&ctx);
        for index in 0..total {
            assert!(registered(&ctx, &uri(index)), "{} was released", uri(index));
        }
    }

    #[test]
    fn images_that_scroll_away_are_released() {
        let ctx = context();
        let total = RESIDENT + 40;
        for index in 0..total {
            include(&ctx, &uri(index), b"image bytes");
        }
        frame(&ctx);
        sweep(&ctx);
        let kept = (0..total)
            .filter(|index| registered(&ctx, &uri(*index)))
            .count();
        assert_eq!(kept, RESIDENT, "the window should hold exactly its budget");
    }

    #[test]
    fn a_released_image_is_registered_again_when_it_comes_back() {
        let ctx = context();
        for index in 0..RESIDENT + 40 {
            include(&ctx, &uri(index), b"image bytes");
        }
        frame(&ctx);
        sweep(&ctx);
        // The oldest URI lost its bytes. Drawing it again must put them back,
        // because egui has no other way to resolve a `bytes://` image.
        include(&ctx, &uri(0), b"image bytes");
        assert!(registered(&ctx, &uri(0)));
    }

    #[test]
    fn images_that_scroll_away_free_their_texture() {
        let ctx = context();
        ctx.add_image_loader(Arc::new(ReadyLoader(Arc::new(egui::ColorImage::example()))));
        let uris: Vec<String> = (0..RESIDENT + 8).map(uri).collect();
        for uri in &uris {
            include(&ctx, uri, b"image bytes");
        }
        let mut showing = draw(&ctx, &uris);
        let shown = allocated(&ctx);
        assert!(
            shown >= uris.len(),
            "a drawn image should own a texture, saw {shown} for {} images",
            uris.len()
        );
        showing.textures_delta.clear();

        // Nothing on screen: the images past the window go, and the frame that
        // released them is the one that tells the painter to delete them.
        let mut empty = draw(&ctx, &[]);
        assert_eq!(
            empty.textures_delta.free.len(),
            8,
            "the swept textures should be freed by the painter"
        );
        assert_eq!(allocated(&ctx), shown - 8);
        empty.textures_delta.clear();
    }

    #[test]
    fn large_offscreen_photos_fit_the_byte_budget_below_the_uri_limit() {
        let ctx = context();
        let size = [2048, 1024];
        ctx.add_image_loader(Arc::new(ReadyLoader(Arc::new(egui::ColorImage::filled(
            size,
            egui::Color32::WHITE,
        )))));
        let uris: Vec<_> = (0..8).map(uri).collect();
        for uri in &uris {
            touch(&ctx, uri);
        }
        drop(draw(&ctx, &uris));

        let after_scrolling = ctx.run_ui(egui::RawInput::default(), |ui| {
            touch(ui.ctx(), &uris[0]);
            ui.add(egui::Image::from_uri(uris[0].clone()));
            sweep(ui.ctx());
        });
        assert_eq!(after_scrolling.textures_delta.free.len(), 4);
        let manager = ctx.tex_manager();
        let textures = manager.read();
        let still_resident: Vec<_> = textures
            .allocated()
            .filter(|(_, texture)| uris.contains(&texture.name))
            .map(|(_, texture)| texture)
            .collect();
        assert!(still_resident.iter().any(|texture| texture.name == uris[0]));
        let offscreen: usize = still_resident
            .into_iter()
            .filter(|texture| texture.name != uris[0])
            .map(egui::epaint::textures::TextureMeta::bytes_used)
            .sum();
        assert_eq!(offscreen, OFFSCREEN_TEXTURE_BYTES);
    }

    #[test]
    fn removing_an_animation_releases_all_of_its_uploaded_frames() {
        let ctx = context();
        ctx.add_image_loader(Arc::new(ReadyLoader(Arc::new(egui::ColorImage::example()))));
        let uri = "file:///sticker.webp";
        for index in 0..3 {
            assert!(matches!(
                ctx.try_load_texture(
                    &format!("{uri}#{index}"),
                    egui::TextureOptions::LINEAR,
                    egui::load::SizeHint::default(),
                ),
                Ok(egui::load::TexturePoll::Ready { .. })
            ));
        }
        let before = allocated(&ctx);
        forget(&ctx, uri);
        assert_eq!(allocated(&ctx), before - 3);
    }
}
