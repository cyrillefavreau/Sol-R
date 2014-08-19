/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include <string>

#include "GPUKernel.h"

//<COLOURSCHEME 0>
//  <COLOUR 0.696078 0.666667 0.372549>
//  <COLOUR 0.176471 0.058824 0.431372>
//  <COLOUR 0.000000 0.529412 0.725490>
//  <COLOUR 0.549020 0.000000 0.500000>

struct MapMaterialScheme
{
   unsigned int index;
   Vertex material[4];
};

// <BLOCK LOCATION 8 0 0 DIMENSION 53 36 0 NORTH 0 SOUTH 0 EAST 0 WEST 0 SPLIT 0 HOLLOW 0>
struct MapBlock
{
   int zone;
   int type;
   Vertex location;
   Vertex dimension;
   int north;
   int south;
   int east;
   int west;
   int split;
   int hollow;
};

// <ZONE LOCATION 0 0 0 DIMENSIONS 54 37 23 COLOURSCHEME 0>
struct MapZone
{
   Vertex location;
   Vertex dimension;
   int    colorScene;
};

// <JEWEL LOCATION 1 6 0>
struct MapJewel
{
   Vertex location;
};

class RAYTRACINGENGINE_API MapReader
{
public:
   MapReader(void);
   ~MapReader(void);

   Vertex loadFromFile( 
      const std::string& filename,
      GPUKernel& cudaKernel);
};
