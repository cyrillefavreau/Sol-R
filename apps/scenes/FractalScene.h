/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include "Scene.h"

class FractalScene : public Scene
{

public:
   FractalScene( const std::string& name, const int nbMaxPrimitivePerBox  );
   ~FractalScene(void);

protected:
   virtual void doInitialize();
   virtual void doAnimate();
   virtual void doAddLights();

private:
   void createFractals( int iteration, int modelId, int mode, int maxIterations, FLOAT4 center, int material, float interval, float radius );

};

