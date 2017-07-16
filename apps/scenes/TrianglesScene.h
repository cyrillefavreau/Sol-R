/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include <vector>
#include "Scene.h"

extern int m_counter;

class TrianglesScene : public Scene
{

public:
   TrianglesScene( const std::string& name, const int nbMaxPrimitivePerBox  );
   ~TrianglesScene(void);

protected:
   virtual void doInitialize();
   virtual void doAnimate();
   virtual void doAddLights();

protected:

      // Animation
   int m_frameIndex;

private:
   std::vector<std::string> m_objFiles;

};

