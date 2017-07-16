/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include "Scene.h"

class PerpetualMotionScene : public Scene
{

public:
   PerpetualMotionScene( const std::string& name, const int nbMaxPrimitivePerBox  );
   ~PerpetualMotionScene(void);

protected:
   virtual void doInitialize();
   virtual void doAnimate();
   virtual void doAddLights();
};
