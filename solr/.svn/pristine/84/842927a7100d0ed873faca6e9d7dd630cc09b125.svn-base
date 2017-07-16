/* 
* Raytracing Engine
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

#include "GPUKernel.h"

class RAYTRACINGENGINE_API FileMarshaller
{
public:

   FileMarshaller(GPUKernel* cudaKernel);
   ~FileMarshaller(void);

public:

   void loadFromFile( const std::string& filename);
   void saveToFile( const std::string& filename);

private:

   void readSceneInfo( const std::string& line );
   void readPrimitive( const std::string& line );
   void readMaterial( const std::string& line );

private:
   
   GPUKernel* m_gpuKernel;

};

