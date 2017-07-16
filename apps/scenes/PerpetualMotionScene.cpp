/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#include <math.h>
#include <stdlib.h>

#include <Logging.h>

#include "PerpetualMotionScene.h"

PerpetualMotionScene::PerpetualMotionScene( const std::string& name, const int nbMaxPrimitivePerBox  )
    : Scene( name, nbMaxPrimitivePerBox )
{
}

PerpetualMotionScene::~PerpetualMotionScene(void)
{
}

/*
________________________________________________________________________________

createPerpetualMotionScene
________________________________________________________________________________
*/
void PerpetualMotionScene::doInitialize()
{
    // Ground
    // m_nbPrimitives = m_gpuKernel->addPrimitive(  ptCheckboard );  m_gpuKernel->setPrimitive( m_nbPrimitives, 0.f, m_groundHeight, 0.f,  50000.f, 0.f, 50000.f, rand()%50+30, 5000, 5000);

    float size = 8000.f;
#if 0
    // Walls
    m_nbPrimitives = m_gpuKernel->addPrimitive(  ptXYPlane );  m_gpuKernel->setPrimitive( m_nbPrimitives, 0.f,    0.f,   size,  size, size, 0.f, 10+rand()%10);

    int m = 10+rand()%10;
    m_nbPrimitives = m_gpuKernel->addPrimitive(  ptYZPlane );  m_gpuKernel->setPrimitive( m_nbPrimitives, -size,     0.f,   6000,  0.f, size, 2000, m);
    m_nbPrimitives = m_gpuKernel->addPrimitive(  ptYZPlane );  m_gpuKernel->setPrimitive( m_nbPrimitives, -size,     0.f,  -6000,  0.f, size, 2000, m);
    m_nbPrimitives = m_gpuKernel->addPrimitive(  ptYZPlane );  m_gpuKernel->setPrimitive( m_nbPrimitives, -size, -2000.f,      0,  0.f, 2000, size, m);
    m_nbPrimitives = m_gpuKernel->addPrimitive(  ptYZPlane );  m_gpuKernel->setPrimitive( m_nbPrimitives, -size,  6000.f,      0,  0.f, 2000, size, m);

    m_nbPrimitives = m_gpuKernel->addPrimitive(  ptYZPlane );  m_gpuKernel->setPrimitive( m_nbPrimitives,  size,     0.f,    0.f,  0.f, size, size, 10+rand()%10);
    m_nbPrimitives = m_gpuKernel->addPrimitive(  ptXZPlane );  m_gpuKernel->setPrimitive( m_nbPrimitives,   0.f,  size,    0.f, size,  0.f, size, 10+rand()%10, 1000, 1000);
#endif // 0

    // Complex object
    float r = rand()%500+20.f;

    int material1=1020; //BASIC_REFLECTION_MATERIAL_004;
    int material2=1021; //BASIC_REFLECTION_MATERIAL_005;
    int material3=1022; //BASIC_REFLECTION_MATERIAL_003;
    int material4=1023; //BASIC_REFLECTION_MATERIAL_002;
    int material5=1024; //BASIC_REFLECTION_MATERIAL_001;
    // Portique
    float radius = 200.f;
    m_nbPrimitives = m_gpuKernel->addPrimitive( ptCylinder, true ); m_gpuKernel->setPrimitive( m_nbPrimitives, -4000, m_groundHeight, -4000,  -2500, 4000, 0, radius, 0.f, 0.f, material1);
    m_nbPrimitives = m_gpuKernel->addPrimitive( ptCylinder, true ); m_gpuKernel->setPrimitive( m_nbPrimitives, -4000, m_groundHeight,  4000,  -2500, 4000, 0, radius, 0.f, 0.f, material1);
    m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere, true );   m_gpuKernel->setPrimitive( m_nbPrimitives, -2500,  4000,     0, radius,    0, 0, material1);
    m_nbPrimitives = m_gpuKernel->addPrimitive( ptCylinder, true ); m_gpuKernel->setPrimitive( m_nbPrimitives, -2500,  4000,     0,   2500, 4000, 0, radius, 0.f, 0.f, material1);
    m_nbPrimitives = m_gpuKernel->addPrimitive( ptCylinder, true ); m_gpuKernel->setPrimitive( m_nbPrimitives,  4000, m_groundHeight, -4000,   2500, 4000, 0, radius, 0.f, 0.f, material1);
    m_nbPrimitives = m_gpuKernel->addPrimitive( ptCylinder, true ); m_gpuKernel->setPrimitive( m_nbPrimitives,  4000, m_groundHeight,  4000,   2500, 4000, 0, radius, 0.f, 0.f, material1);
    m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere, true );   m_gpuKernel->setPrimitive( m_nbPrimitives,  2500,  4000,     0, radius,    0, 0, material1);
    m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere, true );   m_gpuKernel->setPrimitive( m_nbPrimitives, -2500,  4000,     0, radius,    0, 0, material1);

    // Balls
    m_nbPrimitives = m_gpuKernel->addPrimitive( ptCylinder, true ); m_gpuKernel->setPrimitive( m_nbPrimitives,  -2000, -1000.f,   0, -2000, 4000, -200, 10, 0.f, 0.f, material2);
    m_nbPrimitives = m_gpuKernel->addPrimitive( ptCylinder, true ); m_gpuKernel->setPrimitive( m_nbPrimitives,  -2000, -1000.f,   0, -2000, 4000,  200, 10, 0.f, 0.f, material2);
    m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere, true );   m_gpuKernel->setPrimitive( m_nbPrimitives,  -2000, -1000.f,   0,  1000,    0, 0, material5);

    m_nbPrimitives = m_gpuKernel->addPrimitive( ptCylinder, true ); m_gpuKernel->setPrimitive( m_nbPrimitives,      0, -1000.f,     0,     0, 4000,-200, 10, 0.f, 0.f, material2);
    m_nbPrimitives = m_gpuKernel->addPrimitive( ptCylinder, true ); m_gpuKernel->setPrimitive( m_nbPrimitives,      0, -1000.f,     0,     0, 4000, 200, 10, 0.f, 0.f, material2);
    m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere, true );   m_gpuKernel->setPrimitive( m_nbPrimitives,      0, -1000.f,     0,  1000,    0, 0, material3);

    m_nbPrimitives = m_gpuKernel->addPrimitive( ptCylinder, true ); m_gpuKernel->setPrimitive( m_nbPrimitives, 2000, -1000.f,     0,  2000, 4000, -200, 10, 0.f, 0.f, material2);
    m_nbPrimitives = m_gpuKernel->addPrimitive( ptCylinder, true ); m_gpuKernel->setPrimitive( m_nbPrimitives, 2000, -1000.f,     0,  2000, 4000,  200, 10, 0.f, 0.f, material2);
    m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere, true );   m_gpuKernel->setPrimitive( m_nbPrimitives, 2000, -1000.f,     0,  1000,    0, 0, material4);

    m_gpuKernel->getSceneInfo().nbRayIterations = 20;
}

void PerpetualMotionScene::doAnimate()
{
}

void PerpetualMotionScene::doAddLights()
{
    // lights
    LOG_INFO(1,"Adding light");
    float size=1000.f;
    m_nbPrimitives = m_gpuKernel->addPrimitive(  ptXZPlane );  m_gpuKernel->setPrimitive(  m_nbPrimitives, 5000.f, 5000.f, -5000.f,  size, 0.f, size*4.f, DEFAULT_LIGHT_MATERIAL);
}
