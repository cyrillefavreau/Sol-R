/* 
* Copyright (C) 2011-2014 Cyrille Favreau <cyrille_favreau@hotmail.com>
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

// Defines
#ifndef WIN32
#define nullptr 0
#endif

// Oculus Rift
#define USE_OCULUS

// inline   
#define __INLINE__ // inline
#define __ALIGN16__ //__align__(16)

//CUDA Specific
#undef USE_MANAGED_MEMORY

// Raytracer features
#define EXTENDED_GEOMETRY      // Includes spheres, cylinders, etc
#define AUTOFOCUS
#undef  PHOTON_ENERGY
#undef  DODGY_REFRACTIONS

// Consts
const int MAX_GPU_COUNT     = 32;
const int MAX_STREAM_COUNT  = 32;
const int NB_MAX_ITERATIONS = 10;

const long BOUNDING_BOXES_TREE_DEPTH = 64;
const long NB_MAX_BOXES      = 3500000;
const long NB_MAX_PRIMITIVES = 3500000;
const long NB_MAX_LAMPS      = 512;
const long NB_MAX_MATERIALS  = 65506+30; // Last 30 materials are reserved
const long NB_MAX_TEXTURES   = 512;
const long NB_MAX_FRAMES     = 512;
const long NB_MAX_LIGHTINFORMATIONS = 512;
const long MAX_BITMAP_WIDTH  = 1920;
const long MAX_BITMAP_HEIGHT = 1080;

// Constants
#define MATERIAL_NONE -1
#define TEXTURE_NONE  -1
#define TEXTURE_MANDELBROT -2
#define TEXTURE_JULIA -3
#define gColorDepth  3

// Globals
#define PI 3.14159265358979323846f
#define EPSILON 1.f
#define REBOUND_EPSILON 0.00001f

// Skybox matrerials
const int SKYBOX_FRONT_MATERIAL  = NB_MAX_MATERIALS-2;
const int SKYBOX_RIGHT_MATERIAL  = NB_MAX_MATERIALS-3;
const int SKYBOX_BACK_MATERIAL   = NB_MAX_MATERIALS-4;
const int SKYBOX_LEFT_MATERIAL   = NB_MAX_MATERIALS-5;
const int SKYBOX_TOP_MATERIAL    = NB_MAX_MATERIALS-6;
const int SKYBOX_BOTTOM_MATERIAL = NB_MAX_MATERIALS-7;

// Ground material
const int SKYBOX_GROUND_MATERIAL = NB_MAX_MATERIALS-8;

// Cornell Box material
const int CORNELLBOX_FRONT_MATERIAL  = NB_MAX_MATERIALS-9;
const int CORNELLBOX_RIGHT_MATERIAL  = NB_MAX_MATERIALS-10;
const int CORNELLBOX_BACK_MATERIAL   = NB_MAX_MATERIALS-11;
const int CORNELLBOX_LEFT_MATERIAL   = NB_MAX_MATERIALS-12;
const int CORNELLBOX_TOP_MATERIAL    = NB_MAX_MATERIALS-13;
const int CORNELLBOX_BOTTOM_MATERIAL = NB_MAX_MATERIALS-14;
const int CORNELLBOX_GROUND_MATERIAL = NB_MAX_MATERIALS-15;

// Fractals
const int MANDELBROT_MATERIAL = NB_MAX_MATERIALS-16;
const int JULIA_MATERIAL      = NB_MAX_MATERIALS-17;

// Basic reflection materials
const int BASIC_REFLECTION_MATERIAL_001 = NB_MAX_MATERIALS-18;
const int BASIC_REFLECTION_MATERIAL_002 = NB_MAX_MATERIALS-19;
const int BASIC_REFLECTION_MATERIAL_003 = NB_MAX_MATERIALS-20;
const int BASIC_REFLECTION_MATERIAL_004 = NB_MAX_MATERIALS-21;
const int BASIC_REFLECTION_MATERIAL_005 = NB_MAX_MATERIALS-22;
const int BASIC_REFLECTION_MATERIAL_006 = NB_MAX_MATERIALS-23;

// Light source materials
const int LIGHT_MATERIAL_001            = NB_MAX_MATERIALS-24;
const int LIGHT_MATERIAL_002            = NB_MAX_MATERIALS-25;
const int LIGHT_MATERIAL_003            = NB_MAX_MATERIALS-26;
const int LIGHT_MATERIAL_004            = NB_MAX_MATERIALS-27;
const int LIGHT_MATERIAL_005            = NB_MAX_MATERIALS-28;
const int LIGHT_MATERIAL_006            = NB_MAX_MATERIALS-29;
const int LIGHT_MATERIAL_007            = NB_MAX_MATERIALS-30;
const int LIGHT_MATERIAL_008            = NB_MAX_MATERIALS-31;
const int LIGHT_MATERIAL_009            = NB_MAX_MATERIALS-32;
const int LIGHT_MATERIAL_010            = NB_MAX_MATERIALS-33;
const int DEFAULT_LIGHT_MATERIAL        = NB_MAX_MATERIALS-34;

// Basic color materials
const int WHITE_MATERIAL                = NB_MAX_MATERIALS-35;
const int RED_MATERIAL                  = NB_MAX_MATERIALS-36;
const int GREEN_MATERIAL                = NB_MAX_MATERIALS-37;
const int BLUE_MATERIAL                 = NB_MAX_MATERIALS-38;
const int YELLOW_MATERIAL               = NB_MAX_MATERIALS-39;
const int PURPLE_MATERIAL               = NB_MAX_MATERIALS-40;

// Ground material
const int SKYBOX_SPHERE_MATERIAL        = NB_MAX_MATERIALS-41;

#ifdef USE_KINECT
// Kinect
const int KINECT_COLOR_MATERIAL         = NB_MAX_MATERIALS-42;
const int KINECT_DEPTH_MATERIAL         = NB_MAX_MATERIALS-43;

const int KINECT_COLOR_TEXTURE          = 0;
const int KINECT_DEPTH_TEXTURE          = 1;

const int KINECT_COLOR_WIDTH  = 640;
const int KINECT_COLOR_HEIGHT = 480;
const int KINECT_COLOR_DEPTH  = 4;
const int KINECT_COLOR_SIZE   = KINECT_COLOR_WIDTH*KINECT_COLOR_HEIGHT*KINECT_COLOR_DEPTH;

const int KINECT_DEPTH_WIDTH  = 320;
const int KINECT_DEPTH_HEIGHT = 240;
const int KINECT_DEPTH_DEPTH  = 2;
const int KINECT_DEPTH_SIZE   = KINECT_DEPTH_WIDTH*KINECT_DEPTH_HEIGHT*KINECT_DEPTH_DEPTH;

#endif // USE_KINECT
