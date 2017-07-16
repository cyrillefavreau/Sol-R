/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include <vector>
#include "Scene.h"

class AnimationScene : public Scene
{

public:
   AnimationScene( const std::string& name, const int nbMaxPrimitivePerBox  );
   ~AnimationScene(void);

protected:
   virtual void doInitialize();
   virtual void doAnimate();
   virtual void doAddLights();

private:
   std::vector<std::string> m_animations;
   std::string m_fileName;
   int m_currentFrame;
   int m_wait;
   bool m_forward;
};

