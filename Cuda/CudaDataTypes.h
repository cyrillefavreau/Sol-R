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
};

struct Ray
{
   float4 origin;
   float4 direction;
   float4 inv_direction;
   int4   signs;
};

// Enums
enum PrimitiveType 
{
	ptSphere      = 0,
	ptTriangle    = 1,
	ptCheckboard  = 2,
	ptCamera      = 3,
	ptXYPlane     = 4,
	ptYZPlane     = 5,
	ptXZPlane     = 6,
	ptCylinder    = 7,
   ptMagicCarpet = 8,
   ptEnvironment = 9
};

// TODO! Data structure is too big!!!
struct Material
{
	float4 color;
   float4 specular;       // x: value, y: power, w: coef, z: inner illumination
   float4 reflection;     
	float4 refraction;
   float4 transparency;
   float4 bidon;
	int4   textured;
   int4   textureId;
};

struct BoundingBox
{
   float4 parameters[2];
   int1   nbPrimitives;
   int1   startIndex;
   int2   bidon; // Alignment issues
};

struct Primitive
{
	float4 p0;
   /*
	float4 p1;
	float4 p2;
	float4 v0;
	float4 v1;
	float4 v2;
   */
	float4 normal;
	float4 rotation;
	float4 size;
	int1   type;
	int1   materialId;
	float2 materialInfo;
};

// Post processing effect

enum PostProcessingType 
{
   ppe_none,
   ppe_depthOfField,
   ppe_ambientOcclusion,
   ppe_cartoon,
   ppe_antiAliasing
};

struct PostProcessingInfo
{
   int1   type;
   float1 param1; // pointOfFocus;
   float1 param2; // strength;
   int1   param3; // iterations;
};

