## Source, Shadows and Shading

**Surface Brightness:** The surface brightness depends on its albedo (reflection) and amount of illumination it receives.
**Shading Model:** A model of how the brightness of a surface is obtained
- We can interpret pixel value with shading models.
- We can interpret shadows and explain their puzzling

 If two surface patches with the same BRDF see the same incoming hemisphere, then the radiation they output must be the same.  


`Lambert determined` the distribution of 'brightness' on a uniform plane at the base on an infinity high black wall illumindated by an overcast sky. 

Anything that emits light that is internally generated is consider as `light source`.  

$$E(P) = \int_\Omega L_e(P,\theta_0, \phi_0)\cos\theta  dw  $$

**Point Source:** when we assume that the light source is an extremely small sphere, in fact a point is known as point source. The radiosity of a source point is, $ρ(\frac{\epsilon}{r})E\cos\theta$. [Where ρ =surface albedo, E = exitance of the source integrated over the small patch, ε = radius, r = distance]  

- **Nearby Point Source** $ρ_d(P)\frac{N(P).S(P)}{r(P)^2}$ S = source vector, N(P) = The unit normal to the surface, S(P) = a vector from P to the source whoes lenght is ε<sup>2</sup>E. 
- **Infinity Point Source** the radiosity B(P) = ρ<sub>d</sub>(P)(N · S)



# Photometric Stereo 
Reconstruct a path of surface from a series of pictures of teh surface taken under different illumination. 
- Let, an orthographic camera in coordinate system (x,y,z) in space project point(x,y) in the image. 
- To measure the shape of the surface, we need to obtain `depth` to the surface. 
- (x,y, f(x,y)) - Monge patch: We can determine a unique point  on the surface by giving the image coordinates. 
- To obtain a measurement of a solid object, we would need to reconstruct more than one patch. because we need to observe the back of the image. 
-  `Photometric Stereo` is a method for recovering a representation of Monge patch from image data. 
   -  Image intensity values for several different images of surface in a fixed view illuminated by different sources. 
   -  The method recovers the height of the surface at points corresponding to each pixel. Known as `height map` or `depth map`. 
   -  Fix the camera and source in position and illuminate the surface using a point source that is far away compared with the size of the surface. 
   -  The radiosity at a point P on the surface is $B(P)=ρ(P)N(P).S_1$, where N - The unit surface normal, S
   -  