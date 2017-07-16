/* Copyright (c) 2011-2014, Cyrille Favreau
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille_favreau@hotmail.com>
 *
 * This file is part of Sol-R <https://github.com/cyrillefavreau/Sol-R>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef TYPES_H
#define TYPES_H

#ifdef __APPLE__
#include </System/Library/Frameworks/OpenCL.framework/Headers/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include "../Consts.h"

typedef cl_float4 Vertex;
typedef cl_int4 PrimitiveXYIdBuffer;
typedef unsigned char BitmapBuffer;
typedef float RandomBuffer;
typedef int Lamp;
typedef cl_int4 INT4;
typedef cl_int3 INT3;
typedef cl_int2 INT2;
typedef cl_float4 FLOAT4;
typedef cl_float2 FLOAT2;

struct PostProcessingBuffer {
  cl_float4 colorInfo;
  cl_float4 sceneInfo;
};

// 3D vision tqype
enum VisionType {
  vtStandard = 0,
  vtAnaglyph = 1,
  vt3DVision = 2,
  vtFishEye = 3,
  vtVolumeRendering = 4
};

enum GeometryComplexity { gcTrianglesOnly = 0, gcExtendedGeometry = 1 };

enum DoubleSidedTriangles {
  dtDoubleSidedTrianglesOff = 0,
  dtDoubleSidedTrianglesOn = 1
};

enum DraftMode { dmDraftModeOff = 0, dmDraftModeOn = 1 };

enum CameraModes {
  cmStandard = 0,
  cmIsometric3D = 1,
  cmAntialiazing = 2,
  cmVolumeRendering = 3
};

// Bitmap format
enum OutputType {
  otOpenGL = 0, // RGB 24bit
  otDelphi = 1, // BGR 24bit
  otJPEG = 2    // RGB 24bit inverted bitmap
};

enum AdvancedIllumination {
  aiNone = 0,
  aiGlobalIllumination,
  aiAdvancedGlobalIllumination,
  aiRandomIllumination
};

// Scene information
struct SceneInfo {
  cl_int2 size;         // Image size
  cl_int graphicsLevel; // Graphics level( No Shading=0, Lambert=1, Specular=2,
                        // textured=3, Reflections and Refractions=4, Shadows=5
                        // )
  cl_int nbRayIterations; // Maximum number of ray iterations for current frame
  cl_float transparentColor; // Value above which r+g+b color is considered as
                             // transparent
  cl_float viewDistance;     // Maximum viewing distance
  cl_float shadowIntensity;  // Shadow intensity( off=0, pitch black=1)
  cl_float width3DVision;    // 3D: Distance between both eyes
  cl_float4 backgroundColor; // Background color
  cl_int renderingType; // Rendering type( Standard=0, Anaglyph=1, OculusVR=2,
                        // FishEye=3)
  cl_int renderBoxes;   // Activate bounding box rendering( off=0, on=1 );
  cl_int pathTracingIteration;     // Current iteration for current frame
  cl_int maxPathTracingIterations; // Maximum number of iterations for current
                                   // frame
  cl_int4 misc; // x: Bitmap encoding( OpenGL=0, Delphi=1, JPEG=2 )
                // y: Timer
                // z: Fog( 0:disabled, 1:enabled )
                // w: Camera modes( Standard=0, Isometric 3D=1, Antialiazing=2 )
  cl_int4 parameters; // x: Double-sided triangles( 0:disabled, 1:enabled )
                      // y: Extended geometry ( 0:disabled, 1:enabled )
  // z: Advanced features( 0:disabled, 1:global illumination, 2: advanced global
  // illumination, 3: random lightning )
  // w: Draft mode(0:disabled, 1:enabled)
  cl_int4 skybox; // x: size
                  // y: material Id
};

// Ray structure
struct Ray {
  Vertex origin;        // Origin of the ray
  Vertex direction;     // Direction of the ray
  Vertex inv_direction; // Inverted direction( Used for optimal Ray-Box
                        // intersection )
  cl_int4 signs;        // Signs ( Used for optimal Ray-Box intersection )
};

// Light Information Structure used for global illumination
// When iterating on frames, lights sources are randomly picked from an array of
// that
// very structure, in order to simulate global illumination
struct __ALIGN16__ LightInformation {
  cl_int2 attribute; // x: ID of the emitting primitive
                     // y: Material ID
  Vertex location;   // Position in space
  cl_float4 color;   // Light
};

// Primitive types
enum PrimitiveType {
  ptSphere = 0,
  ptCylinder = 1,
  ptTriangle = 2,
  ptCheckboard = 3,
  ptCamera = 4,
  ptXYPlane = 5,
  ptYZPlane = 6,
  ptXZPlane = 7,
  ptMagicCarpet = 8,
  ptEnvironment = 9,
  ptEllipsoid = 10,
  ptQuad = 11,
  ptCone = 12
};

// Material structure
struct __ALIGN16__ Material {
  cl_float4 innerIllumination; // x: Inner illumination
                               // y: Diffusion strength
                               // z: <not used>
                               // w: Illuminance
  cl_float4 color;             // Color( R,G,B )
  cl_float4 specular;          // x: Value
                               // y: Power
                               // z: <not used>
                               // w: <not used>
  cl_float
      reflection; // Reflection rate( No reflection=0 -> Full reflection=1 )
  cl_float refraction;   // Refraction index( ex: glass=1.33 )
  cl_float transparency; // Transparency rate( Opaque=0 -> Full transparency=1 )
  cl_float opacity;      // Opacity strength
  cl_int4 attributes; // x: Fast transparency( off=0, on=1 ). Fast transparency
                      // produces no shadows
  //    and drops intersections if rays intersects primitive with the same
  //    material ID
  // y: Procedural textures( off=0, on=1 )
  // z: Wireframe( off=0, on=1 ). Wire frame produces no shading
  // w: Wireframe Width
  cl_int4 textureMapping;        // x: U padding
                                 // y: V padding
                                 // z: Texture ID (Deprecated)
                                 // w: Texture color depth
  cl_int4 textureOffset;         // x: Offset in the diffuse map
                                 // y: Offset in the normal map
                                 // z: Offset in the bump map
                                 // w: Offset in the specular map
  cl_int4 textureIds;            // x: Diffuse map
                                 // y: Normal map
                                 // z: Bump map
                                 // w: Specular map
  cl_int4 advancedTextureOffset; // x: Reflection map
                                 // y: Transparency map
                                 // z: Ambiant Occulusion
                                 // w: not used
  cl_int4 advancedTextureIds;    // x: Reflection map
                                 // y: Transparency map
                                 // z: Ambiant Occulusion
                                 // w: not used
  cl_float2 mappingOffset; // Texture mapping offsets based on sceneInfo.misc.y
};

// Bounding Box Structure
struct __ALIGN16__ BoundingBox {
  Vertex parameters[2];    // Bottom-Left and Top-Right corners
  cl_int nbPrimitives;     // Number of primitives in the box
  cl_int startIndex;       // Index of the first primitive in the box
  cl_int2 indexForNextBox; // If no intersection, how many of the following
                           // boxes can be skipped?
};

// Primitive Structure
struct __ALIGN16__ Primitive {
  // Vertices
  Vertex p0;
  Vertex p1;
  Vertex p2;
  // Normals
  Vertex n0;
  Vertex n1;
  Vertex n2;
  // Size( x,y,z )
  Vertex size;
  // Type( See PrimitiveType )
  cl_int type;
  // Index
  cl_int index;
  // Material ID
  cl_int materialId;
  // Texture coordinates
  Vertex vt0;
  Vertex vt1;
  Vertex vt2;
};

enum TextureType {
  tex_diffuse,
  tex_bump,
  tex_normal,
  tex_ambient_occlusion,
  tex_reflective,
  tex_specular,
  tex_transparent
};

// Texture information structure
struct __ALIGN16__ TextureInformation {
  unsigned char *buffer; // Pointer to the texture
  int offset;   // Offset of the texture in the global texture buffer (the one
                // that will be transfered to the GPU)
  cl_int3 size; // Size of the texture
  TextureType type; // Texture type (diffuse, normal, bump, etc.)
};

// Post processing types
// Effects are based on the PostProcessingBuffer
enum PostProcessingType {
  ppe_none,             // No effect
  ppe_depthOfField,     // Depth of field
  ppe_ambientOcclusion, // Ambient occlusion
  ppe_radiosity,        // Radiosity
  ppe_filter,           // Various Filters
  ppe_cartoon           // Cartoon
};

// Post processing information
struct PostProcessingInfo {
  cl_int type;     // Type( See PostProcessingType enum )
  cl_float param1; // Parameter role depends on post processing type
  cl_float param2; // Parameter role depends on post processing type
  cl_int param3;   // Parameter role depends on post processing type
};

// Deprecated structures
// Scene information
struct SceneInfo1 {
  cl_int2 size;         // Image size
  cl_int graphicsLevel; // Graphics level( No Shading=0, Lambert=1, Specular=2,
                        // textured=3, Reflections and Refractions=4, Shadows=5
                        // )
  cl_int nbRayIterations; // Maximum number of ray iterations for current frame
  cl_float transparentColor; // Value above which r+g+b color is considered as
                             // transparent
  cl_float viewDistance;     // Maximum viewing distance
  cl_float shadowIntensity;  // Shadow intensity( off=0, pitch black=1)
  cl_float width3DVision;    // 3D: Distance between both eyes
  cl_float4 backgroundColor; // Background color
  cl_int renderingType; // Rendering type( Standard=0, Anaglyph=1, OculusVR=2,
                        // FishEye=3)
  cl_int renderBoxes;   // Activate bounding box rendering( off=0, on=1 );
  cl_int pathTracingIteration;     // Current iteration for current frame
  cl_int maxPathTracingIterations; // Maximum number of iterations for current
                                   // frame
  cl_int4 misc; // x: Bitmap encoding( OpenGL=0, Delphi=1, JPEG=2 )
                // y: Timer
                // z: Fog( 0:disabled, 1:enabled )
                // w: Camera modes( Standard=0, Isometric 3D=1, Antialiazing=2 )
  cl_int4 parameters; // x: Double-sided triangles( 0:disabled, 1:enabled )
                      // y: Gradient background( 0:disabled, 1:enabled )
                      // z: Advanced features( 0:disabled, 1:enabled )
                      // w: Draft mode(0:disabled, 1:enabled)
};

#endif // TYPES_H
