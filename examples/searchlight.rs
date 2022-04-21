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
        eye: glm::vec3(278.0, 273.0, 1.0),
        direction: glm::vec3(0.0, 0.0, 1.0),
        up: glm::vec3(0.0, 1.0, 0.0),
        fov: 2.1,
        ..Default::default()
    };

    let white = Material::diffuse(hex_color(0xAAAAAA));
    let red = Material::diffuse(hex_color(0xBC0000));
    let _yellow = Material::diffuse(hex_color(0xBCBC00));
    let green = Material::diffuse(hex_color(0x00BC00));
    let light_mtl = Material::light(hex_color(0xFFFEFA), 1000.0);

    let floor = polygon(&[
        glm::vec3(0.0, 0.0, 0.0),
        glm::vec3(0.0, 0.0, 559.2),
        glm::vec3(556.0, 0.0, 559.2),
        glm::vec3(556.0, 0.0, 0.0),
    ]);

    let p1 = glm::vec3(343.0, 548.9, 227.0);
    let p2 = glm::vec3(343.0, 548.9, 332.0);
    let p3 = glm::vec3(213.0, 548.9, 332.0);
    let p4 = glm::vec3(213.0, 548.9, 227.0);

    let c1 = glm::vec3(0.0, 548.9, 0.0);
    let c2 = glm::vec3(556.0, 548.9, 0.0);
    let c3 = glm::vec3(556.0, 548.9, 559.2);
    let c4 = glm::vec3(0.0, 548.9, 559.2);

    let br = glm::vec3(p3[0], c4[1], c4[2]);
    let bl = glm::vec3(p2[0], c3[1], c3[2]);
    let fr = glm::vec3(p4[0], c1[1], c1[2]);
    let fl = glm::vec3(p1[0], c2[1], c2[2]);

    let ceiling_1 = polygon(&[c1, fr, br, c4]);
    let ceiling_2 = polygon(&[p3, p2, bl, br]);
    let ceiling_3 = polygon(&[fl, c2, c3, bl]);
    let ceiling_4 = polygon(&[fr, fl, p1, p4]);

    let shift = glm::vec3(0.0, 500.0, 0.0);
    let b1 = p1 + shift;
    let b2 = p2 + shift;
    let b3 = p3 + shift;
    let b4 = p4 + shift;

    let light_rect = polygon(&[b1, b2, b3, b4]);

    let back_wall = polygon(&[
        glm::vec3(0.0, 0.0, 559.2),
        glm::vec3(0.0, 548.9, 559.2),
        glm::vec3(556.0, 548.9, 559.2),
        glm::vec3(556.0, 0.0, 559.2),
    ]);
    let front_wall = polygon(&[
        glm::vec3(0.0, 0.0, 0.0),
        glm::vec3(556.0, 0.0, 0.0),
        glm::vec3(556.0, 548.9, 0.0),
        glm::vec3(0.0, 548.9, 0.0),
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

    scene.add(Object::new(floor).material(white));
    scene.add(Object::new(ceiling_1).material(white));
    scene.add(Object::new(ceiling_2).material(white));
    scene.add(Object::new(ceiling_3).material(white));
    scene.add(Object::new(ceiling_4).material(white));
    scene.add(Object::new(back_wall).material(white));
    scene.add(Object::new(front_wall).material(white));
    scene.add(Object::new(left_wall).material(red));
    scene.add(Object::new(right_wall).material(green));

    scene.add((light_rect, light_mtl));
    scene.environment = Environment::Color(hex_color(0x87CEEB));

    scene.add(Medium::homogeneous_isotropic(0.00001, 0.002)); // foggy

    let mut time = Instant::now();
    fs::create_dir_all("volumetric_results/")?;
    Renderer::new(&scene, camera)
        .width(1024)
        .height(1024)
        .filter(Filter::Box(1))
        .max_bounces(4)
        .num_samples(1000)
        .iterative_render(500, |iteration, buffer| {
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
                    "volumetric_results/output_{:03}.png",
                    iteration - 1
                ))
                .expect("Failed to save image");
            time = Instant::now();
        });

    Ok(())
}