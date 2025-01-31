//! This is an example of a standard Cornell box, for testing volumetric global illumination
//! with participating media
//!
//! Reference: https://www.graphics.cornell.edu/online/box/data.html

use std::fs;
use std::time::Instant;

use rpt::*;

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    let mut scene = Scene::new();

    let camera = Camera {
        eye: glm::vec3(278.0, 273.0, -800.0),
        direction: glm::vec3(0.0, 0.0, 1.0),
        up: glm::vec3(0.0, 1.0, 0.0),
        fov: 0.686,
        ..Default::default()
    };

    let white = Material::diffuse(hex_color(0xAAAAAA));
    let red = Material::diffuse(hex_color(0xBC0000));
    let yellow = Material::diffuse(hex_color(0xBCBC00));
    let green = Material::diffuse(hex_color(0x00BC00));

    let floor = polygon(&[
        glm::vec3(0.0, 0.0, 0.0),
        glm::vec3(0.0, 0.0, 559.2),
        glm::vec3(556.0, 0.0, 559.2),
        glm::vec3(556.0, 0.0, 0.0),
    ]);
    let ceiling = polygon(&[
        glm::vec3(0.0, 548.9, 0.0),
        glm::vec3(556.0, 548.9, 0.0),
        glm::vec3(556.0, 548.9, 559.2),
        glm::vec3(0.0, 548.9, 559.2),
    ]);

    // width 130, depth 105
    let light_rect = polygon(&[
        glm::vec3(330.0, 548.8, 240.0),
        glm::vec3(330.0, 548.8, 319.0),
        glm::vec3(226.0, 548.8, 319.0),
        glm::vec3(226.0, 548.8, 240.0),
    ]);

    let height = 140.;
    let depth = 105.;
    let width = 130.;
    let center = glm::vec3(213. + 65., 548., 227. + 55.);
    let front_offset = center + glm::vec3(0., 0., depth / 2.);
    let left_offset = center + glm::vec3(-width / 2., 0., 0.);
    let back_offset = center + glm::vec3(0., 0., -depth / 2.);
    let right_offset = center + glm::vec3(width / 2., 0., 0.);

    let off_axis_size = 10.;
    let front_shade = cube()
        .scale(&glm::vec3(130. + off_axis_size * 2., height, off_axis_size))
        .translate(&front_offset);
    let left_shade = cube()
        .scale(&glm::vec3(off_axis_size, height, 105. + off_axis_size * 2.))
        .translate(&left_offset);
    let back_shade = cube()
        .scale(&glm::vec3(130. + off_axis_size * 2., height, off_axis_size))
        .translate(&back_offset);
    let right_shade = cube()
        .scale(&glm::vec3(off_axis_size, height, 105. + off_axis_size * 2.))
        .translate(&right_offset);

    let back_wall = polygon(&[
        glm::vec3(0.0, 0.0, 559.2),
        glm::vec3(0.0, 548.9, 559.2),
        glm::vec3(556.0, 548.9, 559.2),
        glm::vec3(556.0, 0.0, 559.2),
    ]);
    let right_wall = polygon(&[
        glm::vec3(0.0, 0.0, 0.0),
        glm::vec3(0.0, 548.9, 0.0),
        glm::vec3(0.0, 548.9, 559.2),
        glm::vec3(0.0, 0.0, 559.2),
    ]);
    let left_wall = polygon(&[
        glm::vec3(556.0, 0.0, 0.0),
        glm::vec3(556.0, 0.0, 559.2),
        glm::vec3(556.0, 548.9, 559.2),
        glm::vec3(556.0, 548.9, 0.0),
    ]);

    let large_box = cube()
        .scale(&glm::vec3(165.0, 330.0, 165.0))
        .rotate_y(glm::two_pi::<f64>() * (-253.0 / 360.0))
        .translate(&glm::vec3(368.0, 165.0, 351.0));
    let small_box = cube()
        .scale(&glm::vec3(165.0, 165.0, 165.0))
        .rotate_y(glm::two_pi::<f64>() * (-197.0 / 360.0))
        .translate(&glm::vec3(185.0, 82.5, 169.0));

    scene.add(Object::new(floor).material(white));
    scene.add(Object::new(ceiling).material(white));
    scene.add(Object::new(back_wall).material(white));
    scene.add(Object::new(left_wall).material(red));
    scene.add(Object::new(right_wall).material(green));
    scene.add(Object::new(large_box).material(white));
    scene.add(Object::new(small_box).material(white));

    scene.add(Object::new(right_shade).material(yellow));
    scene.add(Object::new(left_shade).material(yellow));
    scene.add(Object::new(front_shade).material(yellow));
    scene.add(Object::new(back_shade).material(yellow));

    let absorb = 0.00005;
    let scat = 0.003;
    let bounce = 10;
    let size = 128;
    let sample = 1000;
    let every_x = 100;
    let watts = 150.0;

    let light_mtl = Material::light(hex_color(0xFFFEFA), watts); // 6500 K
    scene.add((light_rect, light_mtl));

    scene.add(Medium::homogeneous_isotropic(absorb, scat)); // foggy

    fs::create_dir_all("lampshade/pathtrace")?;
    let mut time = Instant::now();
    Renderer::new(&scene, camera)
        .width(size)
        .height(size)
        .max_bounces(bounce)
        .num_samples(sample)
        .iterative_render(every_x, |iteration, buffer| {
            let millis = time.elapsed().as_millis();
            println!(
                "Finished iteration {}, took {} ms, variance: {}",
                iteration,
                millis,
                buffer.variance()
            );
            buffer
                .image()
                .save(format!(
                    "lampshade/pathtrace/output_{:03}.png",
                    iteration - 1
                ))
                .expect("Failed to save image");
            time = Instant::now();
        });

    Ok(())
}
