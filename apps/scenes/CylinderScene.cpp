/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#include <math.h>
#include <stdlib.h>

#include "CylinderScene.h"

int mappedSphere;
Vertex vt0,vt1,vt2;

CylinderScene::CylinderScene( const std::string& name, const int nbMaxPrimitivePerBox  )
 : Scene( name, nbMaxPrimitivePerBox )
{
}

CylinderScene::~CylinderScene(void)
{
}

/*
________________________________________________________________________________

Create simple scene for box validation
________________________________________________________________________________
*/
void CylinderScene::doInitialize()
{
#if 0
   int size=20000;
   int sphereRadius=size/300;
   int cylinderRadius=size/200;
   int material=20;
   int nbSpheres=100000;
   int nbLinks=0;
   std::vector<Vertex> vertices;
   for( int i(0);i<nbSpheres;++i )
   {
      Vertex v;
      v.x = static_cast<float>(rand()%size-size/2);
      v.y = static_cast<float>(rand()%size-size/2);
      v.z = static_cast<float>(rand()%size-size/2);
      vertices.push_back(v);
      m_nbPrimitives=m_gpuKernel->addPrimitive(ptSphere);m_gpuKernel->setPrimitive(m_nbPrimitives,v.x,v.y,v.z,sphereRadius,0.f,0.f,20);
   }

   for( int i(0);i<nbLinks;++i)
   {
      int i1=rand()%vertices.size();
      Vertex v1=vertices[i1];
      int i2=rand()%vertices.size();
      Vertex v2=vertices[i2];
      m_nbPrimitives=m_gpuKernel->addPrimitive(ptCylinder);m_gpuKernel->setPrimitive(m_nbPrimitives,v1.x,v1.y,v1.z,v2.x,v2.y,v2.z,cylinderRadius,0.f,0.f,21);
   }
#else
	float size = 400.f;
   m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere  ); m_gpuKernel->setPrimitive( m_nbPrimitives, 2500, 0, 0, 1000, 0.f, 0.f, 55);
   vt0.x = 0.f;
   vt0.y = 0.f;
   vt0.z = 0.f;
   vt1.x = 10.f;
   vt1.y = 10.f;
   vt1.z = 0.f;
   m_gpuKernel->setPrimitiveTextureCoordinates(m_nbPrimitives,vt0,vt1,vt2);
   mappedSphere=m_nbPrimitives;
   m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere  ); m_gpuKernel->setPrimitive( m_nbPrimitives,-2500, 0, 0, 1000.f, 0.f, 0.f, 45); 
   m_gpuKernel->setPrimitiveTextureCoordinates(m_nbPrimitives,vt0,vt1,vt2);

   // Cylinders
	m_nbPrimitives = m_gpuKernel->addPrimitive( ptCylinder); m_gpuKernel->setPrimitive( m_nbPrimitives,-1000,  1000, 0, 1000, -1000, 0, size, size, 0.f, rand()%100); 
	m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere  ); m_gpuKernel->setPrimitive( m_nbPrimitives,-1000, -1000, 0, size, 0.f, 0.f, rand()%40); 
   m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere  ); m_gpuKernel->setPrimitive( m_nbPrimitives, 1000,  1000, 0, size, 0.f, 0.f, LIGHT_MATERIAL_001 /*rand()%40*/); 

	m_nbPrimitives = m_gpuKernel->addPrimitive( ptCylinder); m_gpuKernel->setPrimitive( m_nbPrimitives,-1000, -1000, 0, 1000, 1000, 0, size, size, 0.f, rand()%100); 
	m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere  ); m_gpuKernel->setPrimitive( m_nbPrimitives,-1000,  1000, 0, size, 0.f, 0.f, rand()%40); 
   m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere  ); m_gpuKernel->setPrimitive( m_nbPrimitives, 1000, -1000, 0, size, 0.f, 0.f, LIGHT_MATERIAL_002 /*rand()%40*/); 
#endif // 1

   m_gpuKernel->getSceneInfo().nbRayIterations = 20;
}

void CylinderScene::doAnimate()
{
   m_rotationAngles.y = 0.01f;
   m_gpuKernel->rotatePrimitives( m_rotationCenter, m_rotationAngles );
   vt0.x = -PI/2.f;
   vt0.y -= 0.05f;
   vt0.z -= 0.05f;
   m_gpuKernel->setPrimitiveTextureCoordinates(mappedSphere,vt0,vt1,vt2);
   m_gpuKernel->setPrimitiveTextureCoordinates(mappedSphere+1,vt0,vt1,vt2);
   m_gpuKernel->compactBoxes(false);
}

void CylinderScene::doAddLights()
{
	// lights
	m_nbPrimitives = m_gpuKernel->addPrimitive(ptSphere);  
   m_gpuKernel->setPrimitive( m_nbPrimitives, 0.f, 10000.f, -10000.f, 2000.f, 0.f, 0, DEFAULT_LIGHT_MATERIAL); 
   m_gpuKernel->setPrimitiveIsMovable(m_nbPrimitives,false);
}
