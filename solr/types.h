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

#include <defines.h>
#include "Consts.h"
#include <vector>
#include <map>

#ifdef USE_OPENCL
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include </System/Library/Frameworks/OpenCL.framework/Headers/opencl.h>
#else // __APPLE__
#include <CL/cl.h>
#endif // __APPLE__

typedef cl_float2 vec2f;
typedef cl_float3 vec3f;
typedef cl_float4 vec4f;
typedef cl_int2 vec2i;
typedef cl_int3 vec3i;
typedef cl_int4 vec4i;
typedef cl_int4 PrimitiveXYIdBuffer;

#define __ALIGN16__
#else
#include <vector_types.h>

typedef float2 vec2f;
typedef int2 vec2i;
typedef float4 vec3f; // In order to align to OpenCL structures, vec3f is a float4 and not a float3. See OpenCL
typedef int4 vec3i;   // documentation for more details. cl_int3 and cl_float3 are identical in size, alignment and
                      // behavior to cl_int4 and cl_float4. See section 6.1.5.
typedef int4 vec4i;
typedef float4 vec4f;
typedef int4 PrimitiveXYIdBuffer;

#define __ALIGN16__ __align__(16)
#endif

// Vectors
typedef std::vector<vec2f> vec2fs;
typedef std::vector<vec3f> vec3fs;
typedef std::vector<vec4f> vec4fs;
typedef std::vector<vec2i> vec2is;
typedef std::vector<vec3i> vec3is;
typedef std::vector<vec4i> vec4is;

typedef unsigned char BitmapBuffer;
typedef float RandomBuffer;
typedef int Lamp;

#define _CRT_SECURE_NO_WARNINGS
#define __INLINE__ inline

struct PostProcessingBuffer
{
    vec4f colorInfo;
    vec4f sceneInfo;
};

// 3D vision tqype
enum VisionType
{
    vtStandard = 0,
    vtAnaglyph = 1,
    vt3DVision = 2,
    vtFishEye = 3,
    vtVolumeRendering = 4
};

enum GeometryComplexity
{
    gcTrianglesOnly = 0,
    gcExtendedGeometry = 1
};

enum DoubleSidedTriangles
{
    dtDoubleSidedTrianglesOff = 0,
    dtDoubleSidedTrianglesOn = 1
};

enum DraftMode
{
    dmDraftModeOff = 0,
    dmDraftModeOn = 1
};

enum CameraModes
{
    cmStandard = 0,
    cmIsometric3D = 1,
    cmAntialiazing = 2,
    cmVolumeRendering = 3
};

// Bitmap format
enum OutputType
{
    otOpenGL = 0, // RGB 24bit
    otDelphi = 1, // BGR 24bit
    otJPEG = 2    // RGB 24bit inverted bitmap
};

enum AdvancedIllumination
{
    aiNone = 0,
    aiGlobalIllumination,
    aiAdvancedGlobalIllumination,
    aiRandomIllumination
};

// Scene information
struct __ALIGN16__ SceneInfo
{
    vec2i size;                   // Image size
    int graphicsLevel;            // Graphics level( No Shading=0, Lambert=1, Specular=2, textured=3, Reflections and
                                  // Refractions=4, Shadows=5)
    int nbRayIterations;          // Maximum number of ray iterations for current frame
    float transparentColor;       // Value above which r+g+b color is considered as transparent
    float viewDistance;           // Maximum viewing distance
    float shadowIntensity;        // Shadow intensity( off=0, pitch black=1)
    float width3DVision;          // 3D: Distance between both eyes
    vec4f backgroundColor;        // Background color
    int renderingType;            // Rendering type( Standard=0, Anaglyph=1, OculusVR=2, FishEye=3)
    int renderBoxes;              // Activate bounding box rendering( off=0, on=1 );
    int pathTracingIteration;     // Current iteration for current frame
    int maxPathTracingIterations; // Maximum number of iterations for current
                                  // frame
    vec4i misc;                   // x: Bitmap encoding( OpenGL=0, Delphi=1, JPEG=2 )
                                  // y: Timer
                                  // z: Fog( 0:disabled, 1:enabled )
                                  // w: Camera modes( Standard=0, Isometric 3D=1, Antialiazing=2 )
    vec4i parameters;             // x: Double-sided triangles( 0:disabled, 1:enabled )
                                  // y: Extended geometry ( 0:disabled, 1:enabled )
                                  // z: Advanced features( 0:disabled, 1:global illumination, 2: advanced global
                                  // illumination, 3: random lightning )
                                  // w: Draft mode(0:disabled, 1:enabled)
    vec4i skybox;                 // x: size
                                  // y: material Id
                                  // z and w: currently unused
};

// Ray structure
struct __ALIGN16__ Ray
{
    vec3f origin;        // Origin of the ray
    vec3f direction;     // Direction of the ray
    vec3f inv_direction; // Inverted direction( Used for optimal Ray-Box
                         // intersection )
    vec4i signs;         // Signs ( Used for optimal Ray-Box intersection )
};

// Light Information Structure used for global illumination
// When iterating on frames, lights sources are randomly picked from an array of
// that very structure, in order to simulate global illumination
struct __ALIGN16__ LightInformation
{
    vec2i attribute; // x: ID of the emitting primitive, y: Material ID
    vec3f location;  // Position in space
    vec4f color;     // Light
};

// Primitive types
enum PrimitiveType
{
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
struct __ALIGN16__ Material
{
    vec4f innerIllumination; // x: Inner illumination
                             // y: Diffusion strength
                             // z: <not used>
                             // w: Illuminance
    vec4f color;             // Color( R,G,B )
    vec4f specular;          // x: Value
                             // y: Power
                             // z: <not used>
                             // w: <not used>
    float reflection;        // Reflection rate( No reflection=0 -> Full reflection=1 )
    float refraction;        // Refraction index( ex: glass=1.33 )
    float transparency;      // Transparency rate( Opaque=0 -> Full transparency=1 )
    float opacity;           // Opacity strength
    vec4i attributes;        // x: Fast transparency( off=0, on=1 ). Fast transparency
    // produces no shadows and drops intersections if rays intersects primitive with the same material ID
    // y: Procedural textures( off=0, on=1 )
    // z: Wireframe( off=0, on=1 ). Wire frame produces no shading
    // w: Wireframe Width
    vec4i textureMapping;        // x: U padding
                                 // y: V padding
                                 // z: Texture ID (Deprecated)
                                 // w: Texture color depth
    vec4i textureOffset;         // x: Offset in the diffuse map
                                 // y: Offset in the normal map
                                 // z: Offset in the bump map
                                 // w: Offset in the specular map
    vec4i textureIds;            // x: Diffuse map
                                 // y: Normal map
                                 // z: Bump map
                                 // w: Specular map
    vec4i advancedTextureOffset; // x: Reflection map
                                 // y: Transparency map
                                 // z: Ambiant Occulusion
                                 // w: not used
    vec4i advancedTextureIds;    // x: Reflection map
                                 // y: Transparency map
                                 // z: Ambiant Occulusion
                                 // w: not used
    vec2f mappingOffset;         // Texture mapping offsets based on sceneInfo.misc.y
};

// Bounding Box Structure
struct __ALIGN16__ BoundingBox
{
    vec3f parameters[2];   // Bottom-Left and Top-Right corners
    int nbPrimitives;      // Number of primitives in the box
    int startIndex;        // Index of the first primitive in the box
    vec2i indexForNextBox; // If no intersection, how many of the following boxes can be skipped?
};
typedef std::map<size_t, BoundingBox> BoundingBoxes;

// Primitive Structure
struct __ALIGN16__ Primitive
{
    // Vertices
    vec3f p0;
    vec3f p1;
    vec3f p2;
    // Normals
    vec3f n0;
    vec3f n1;
    vec3f n2;
    // Size( x,y,z )
    vec3f size;
    // Type( See PrimitiveType )
    int type;
    // Index
    int index;
    // Material ID
    int materialId;
    // Texture coordinates
    vec3f vt0;
    vec3f vt1;
    vec3f vt2;
};
typedef std::map<size_t, Primitive> Primitives;

enum TextureType
{
    tex_diffuse,
    tex_bump,
    tex_normal,
    tex_ambient_occlusion,
    tex_reflective,
    tex_specular,
    tex_transparent
};

// Texture information structure
struct __ALIGN16__ TextureInformation
{
    unsigned char *buffer; // Pointer to the texture
    int offset;            // Offset of the texture in the global texture buffer (the one
                           // that will be transfered to the GPU)
    vec3i size;            // Size of the texture
    TextureType type;      // Texture type (diffuse, normal, bump, etc.)
};

// Post processing types
// Effects are based on the PostProcessingBuffer
enum PostProcessingType
{
    ppe_none,             // No effect
    ppe_depthOfField,     // Depth of field
    ppe_ambientOcclusion, // Ambient occlusion
    ppe_radiosity,        // Radiosity
    ppe_filter,           // Various Filters
    ppe_cartoon           // Cartoon
};

// Post processing information
struct PostProcessingInfo
{
    int type;     // Type( See PostProcessingType enum )
    float param1; // Parameter role depends on post processing type
    float param2; // Parameter role depends on post processing type
    int param3;   // Parameter role depends on post processing type
};

#endif // TYPES_H
