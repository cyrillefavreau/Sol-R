/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include "Scene.h"
   
class WaterScene : public Scene
{
public:
   WaterScene( const std::string& name, const int nbMaxPrimitivePerBox  );
   ~WaterScene();

protected:
   virtual void doInitialize();
   virtual void doAnimate();
   virtual void doAddLights();

private:
   Vertex F( float x, float z, float stepX, float stepZ );
   Vertex P( float t, float stepX, float stepZ );
   void  processCurve( bool update );
   void  processParametricCurve( bool update );

private:
   int     m_material;
   Vertex  m_scale;
   int     m_initialIndex;
   Vertex  m_objectSize;
};

