//! This is an example of a standard Cornell box, for testing volumetric global illumination
//! with participating media
//!
//! Reference: https://www.graphics.cornell.edu/online/box/data.html

use std::fs::{self, File};
use std::time::Instant;

use rpt::*;

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    let mut scene = Scene::new();

    let camera = Camera {
        eye: glm::vec3(0., 200., -500.),
        direction: glm::vec3(0.0, 0.0, 1.0),
        up: glm::vec3(0.0, 1.0, 0.0),
        fov: 0.686,
        ..Default::default()
    };

    //       / \
    //        O
    //  200  | |
    //       | |
    //       | |
    //       | |
    // --------------
    //               |
    //               |
    //               |
    //
    let white = Material::diffuse(hex_color(0xAAAAAA));
    let red = Material::diffuse(hex_color(0xBC0000));
    let yellow = Material::diffuse(hex_color(0xBCBC00));
    let green = Material::diffuse(hex_color(0x00BC00));
    let light_mtl = Material::light(hex_color(0xFFFEFA), 120.0); // 6500 K

    let cylinder = load_obj(File::open("examples/cylinder.obj")?)?;
    let pyramid = load_obj(File::open("examples/pyramid.obj")?)?;

    let sealevel = 0.;
    let rock_height = 100.;
    let lighthouse_base_size = 50.;
    let lighthouse_light_size = 10.;
    let lighthouse_top_size = 10.;

    let lighthouse_x = 100.;

    let rock_pos = glm::vec3(100., sealevel + rock_height / 2., 0.);
    let lighthouse_base_pos = glm::vec3(
        lighthouse_x,
        sealevel + rock_height + lighthouse_base_size / 2.,
        0.,
    );
    let lighthouse_light_pos = glm::vec3(
        lighthouse_x,
        sealevel + rock_height + lighthouse_base_size + lighthouse_light_size / 2.,
        0.,
    );
    let lighthouse_top_pos = glm::vec3(
        lighthouse_x,
        sealevel
            + rock_height
            + lighthouse_base_size
            + lighthouse_light_size
            + lighthouse_top_size / 2.,
        0.,
    );

    let lighthouse_blocker_size = 40.;

    let rocks = cube()
        .scale(&glm::vec3(200., 100., 100.))
        .translate(&rock_pos);

    let rocks = load_obj(File::open("examples/Rock.obj")?)?
        .scale(&glm::vec3(200., 100., 100.))
        .translate(&rock_pos);

    let lighthouse_base1 = cylinder
        .clone()
        .scale(&glm::vec3(10., 50., 10.))
        .translate(&(lighthouse_base_pos - glm::vec3(-30., lighthouse_base_size / 3., 0.)));
    let lighthouse_base2 = cube()
        .clone()
        .scale(&glm::vec3(10., 50., 10.))
        .translate(&lighthouse_base_pos);
    let lighthouse_base3 = cylinder
        .clone()
        .scale(&(glm::vec3(10., 50., 10.) + glm::vec3(30., lighthouse_base_size / 3., 0.)))
        .translate(&lighthouse_base_pos);

    let lighthouse_light = cube()
        .scale(&glm::vec3(5., 5., 5.))
        .translate(&lighthouse_light_pos);

    let lighthouse_light_front = cube()
        .scale(&glm::vec3(
            lighthouse_blocker_size,
            lighthouse_blocker_size,
            5.,
        ))
        .translate(&(lighthouse_light_pos + glm::vec3(0., 5., -13.)));
    let lighthouse_light_back = cube()
        .scale(&glm::vec3(
            lighthouse_blocker_size,
            lighthouse_blocker_size,
            5.,
        ))
        .translate(&(lighthouse_light_pos + glm::vec3(0., 5., 13.)));
    let lighthouse_top = pyramid
        .scale(&glm::vec3(
            lighthouse_blocker_size,
            5.,
            lighthouse_blocker_size,
        ))
        .translate(&(lighthouse_top_pos + glm::vec3(0., 13., 0.)));

    let left_boundary = cube()
        .scale(&glm::vec3(10., 400., 10.))
        .translate(&glm::vec3(250., 0., 0.));
    let right_boundary = cube()
        .scale(&glm::vec3(10., -400., 10.))
        .translate(&glm::vec3(250., 0., 0.));

    scene.add(Object::new(rocks).material(white));
    // scene.add(Object::new(lighthouse_base1).material(white));
    scene.add(Object::new(lighthouse_base2).material(red));
    // scene.add(Object::new(lighthouse_base3).material(white));
    scene.add(Object::new(lighthouse_light_front).material(yellow));
    scene.add(Object::new(lighthouse_light_back).material(yellow));
    scene.add(Object::new(lighthouse_top).material(red));
    scene.add(Object::new(left_boundary).material(green));
    scene.add(Object::new(right_boundary).material(red));

    // scene.add((lighthouse_light, light_mtl));
    scene.add(Light::Point(
        Color::new(1., 1., 1.),
        glm::vec3(0., 200., 0.),
    ));

    let absorb = 0.0008;
    let scat = 0.0008;
    let size = 512;
    let bounce = 10;
    let sample = 100;
    let watts = 1_000_000.;
    let photons = 500_000;

    let gather_size = 100;
    let gather_size_volume = 30;

    // scene.add(Medium::homogeneous_isotropic(absorb, scat)); // foggy

    let image = Renderer::new(&scene, camera)
        .width(size)
        .height(size)
        .max_bounces(bounce)
        .num_samples(sample)
        .gather_size(gather_size)
        .watts(watts)
        .gather_size_volume(gather_size_volume)
        .render();

    image
        .save(format!(
            "vpm/lighthouse/e_{}_{}_{}_{}_{}_{}_{}_{}_{}.png",
            size, bounce, sample, photons, watts, gather_size, gather_size_volume, absorb, scat
        ))
        .expect("Failed to save image");
    Ok(())
}
