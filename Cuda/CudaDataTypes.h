/* 
* Protein Visualizer
* Copyright (C) 2011-2012 Cyrille Favreau <cyrille_favreau@hotmail.com>
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Library General Public
* License as published by the Free Software Foundation; either
* version 2 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Library General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>. 
*/

/*
* Author: Cyrille Favreau <cyrille_favreau@hotmail.com>
*
*/

#pragma once

#include <vector_types.h>

// 3D vision type
enum VisionType
{
   vtStandard = 0,
   vtAnaglyph = 1,
   vt3DVision = 2
};

enum OutputType
{
   otOpenGL = 0,
   otDelphi = 1,
   otJPEG   = 2
};

// Scene information
struct SceneInfo
{
   int1    width;
   int1    height;
   int1    shadowsEnabled;
   int1    nbRayIterations;
   float1  transparentColor;
   float1  viewDistance;
   float1  shadowIntensity;
   float1  width3DVision;
   float4  backgroundColor;
   int1    supportFor3DVision;
   int1    renderBoxes;
   int1    pathTracingIteration;
   int1    maxPathTracingIterations;
   int4    misc; // x : OpenGL=0, Delphi=1, JPEG=2, y: timer, z: fog (0: disabled, 1: enabled), w: 1: Isometric 3D, 2: Antializing
};

struct Ray
{
   float3 origin;
   float3 direction;
   float3 inv_direction;
   int4   signs;
};

// Enums
enum PrimitiveType 
{
	ptSphere      = 0,
	ptCylinder    = 1,
	ptTriangle    = 2,
	ptCheckboard  = 3,
	ptCamera      = 4,
	ptXYPlane     = 5,
	ptYZPlane     = 6,
	ptXZPlane     = 7,
	ptMagicCarpet = 8,
	ptEnvironment = 9,
	ptEllipsoid   = 10
};

// TODO! Data structure is too big!!!
struct Material
{
   float4 innerIllumination; // x: inner illumination, y: diffusion strength
	float4 color;             // w: noise
   float4 specular;          // x: value, y: power, w: coef
   float1 reflection;     
	float1 refraction;
   float1 transparency;
   int1   fastTransparency;
	int4   textureInfo;       // x: procedural, y: textureId, z: wireframe, w: wireframeWidth
};

struct BoundingBox
{
   float3 parameters[2];
   int1   nbPrimitives;
   int1   startIndex;
   int2   type; // Alignment issues
};

struct Primitive
{
	float3 p0;
	float3 p1;
	float3 p2;
	float3 n0;
	float3 n1;
	float3 n2;
	float3 size;
	int1   type;
   int1   index;
	int1   materialId;
	float2 materialInfo;
};

// Post processing effect

enum PostProcessingType 
{
   ppe_none,
   ppe_depthOfField,
   ppe_ambientOcclusion,
   ppe_cartoon
};

struct PostProcessingInfo
{
   int1   type;
   float1 param1; // pointOfFocus;
   float1 param2; // strength;
   int1   param3; // iterations;
};
