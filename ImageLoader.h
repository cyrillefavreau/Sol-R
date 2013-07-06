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

