/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include <map>

#include "Cuda/CudaDataTypes.h"

typedef struct
{
    unsigned char imageTypeCode;
    short int imageWidth;
    short int imageHeight;
    unsigned char bitCount;
} TGAFILE;

class ImageLoader
{
public:
   ImageLoader(void);
   ~ImageLoader(void);

public:
   // BITMAP
   bool loadBMP24(const int index, const std::string& filename, TextureInformation* textureInformations);

   // JPEG
   // https://code.google.com/p/jpeg-compressor
   bool loadJPEG(const int index, const std::string& filename, TextureInformation* textureInformations);

   // TGA
   bool loadTGA(const int index, const std::string& filename, TextureInformation* textureInformations);

};

