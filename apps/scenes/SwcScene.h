/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include <io/SWCReader.h>

#include "Scene.h"

class SwcScene: public Scene
{

public:
   SwcScene( const std::string& name, const int nbMaxPrimitivePerBox  );
   ~SwcScene(void);

protected:

   virtual void doInitialize();
   virtual void doAnimate();
   virtual void doAddLights();

private:

   Morphologies m_morphologies;
   int m_counter;
   int m_previousMaterial;
   int m_previousPrimitiveId;
};

