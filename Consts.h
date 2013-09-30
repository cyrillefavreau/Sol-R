/* 
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

#define USE_OCULUS

// Defines
// #define PHOTON_ENERGY
#define GRADIANT_BACKGROUND
#define EXTENDED_GEOMETRY      // Includes spheres, cylinders, etc
#undef  ADVANCED_FEATURES
#define AUTOFOCUS
#undef  PHOTON_ENERGY
#undef  DODGY_REFRACTIONS

// Consts
const int MAX_GPU_COUNT     = 32;
const int MAX_STREAM_COUNT  = 32;
const int NB_MAX_ITERATIONS = 20;

const int NB_MAX_BOXES      = 4096;
const int NB_MAX_PRIMITIVES = 1000000;
const int NB_MAX_LAMPS      = 10;
const int NB_MAX_MATERIALS  = 65506+30; // Last 30 materials are reserved
const int NB_MAX_TEXTURES   = 1000;
const int NB_MAX_FRAMES     = 1000;
const int NB_MAX_LIGHTINFORMATIONS = 500;


// Constants
#define MATERIAL_NONE -1
#define TEXTURE_NONE  -1
#define TEXTURE_MANDELBROT -2
#define TEXTURE_JULIA -3
#define gColorDepth  3

// Globals
#define PI 3.14159265358979323846f
#define EPSILON 1.f

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
#endif // USE_KINECT
