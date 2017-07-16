/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include "GPUKernel.h"

class RAYTRACINGENGINE_API FileMarshaller
{
public:

   FileMarshaller();
   ~FileMarshaller();

public:

   Vertex loadFromFile( GPUKernel& kernel, const std::string& filename, const Vertex& center, const float scale);
   void saveToFile    ( GPUKernel& kernel, const std::string& filename);
};

