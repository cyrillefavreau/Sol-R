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
   bool loadBMP24(const int index, const std::string& filename, TextureInformation* textureInformations, std::map<int,std::string>& textureFilenames);

   // JPEG
   // https://code.google.com/p/jpeg-compressor
   bool loadJPEG(const int index, const std::string& filename, TextureInformation* textureInformations, std::map<int,std::string>& textureFilenames);

   // TGA
   bool loadTGA(const int index, const std::string& filename, TextureInformation* textureInformations, std::map<int,std::string>& textureFilenames);


protected:
   void processOffset( TextureInformation* textureInformations );

};

