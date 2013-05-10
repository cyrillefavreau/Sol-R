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

const int NB_MAX_BOXES      = 4096;
const int NB_MAX_PRIMITIVES = 500000;
const int NB_MAX_LAMPS      = 10;
const int NB_MAX_MATERIALS  = 130; // Last 30 materials are reserved
const int NB_MAX_TEXTURES   = 40;


// Constants
#define MATERIAL_NONE -1
#define TEXTURE_NONE  -1
#define TEXTURE_MANDELBROT -2
#define TEXTURE_JULIA -3
#define gColorDepth  3

// Consts
const int MAX_GPU_COUNT = 32;
const int NB_MAX_ITERATIONS = 10;

// Globals
#define M_PI 3.14159265358979323846
#define EPSILON 10.f

// Textures
const int gTextureWidth  = 1024;
const int gTextureHeight = 1024;
const int gTextureDepth  = 3;
const int gTextureSize   = gTextureWidth*gTextureHeight*gTextureDepth;

#ifdef USE_KINECT
// Kinect
const int gKinectVideoWidth  = 640;
const int gKinectVideoHeight = 480;
const int gKinectVideo       = 4;
const int gKinectVideoSize   = gKinectVideoWidth*gKinectVideoHeight*gKinectVideo;

const int gKinectDepthWidth  = 320;
const int gKinectDepthHeight = 240;
const int gKinectDepth       = 2;
const int gKinectDepthSize   = gKinectDepthWidth*gKinectDepthHeight*gKinectDepth;

const int gTextureOffset = gKinectVideoSize+gKinectDepthSize;
#else
const int gTextureOffset = 0;
#endif // USE_KINECT
