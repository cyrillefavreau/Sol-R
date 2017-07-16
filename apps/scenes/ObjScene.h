/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include <vector>
#include "Scene.h"

class ObjScene : public Scene
{

public:
   ObjScene( const std::string& name, const int nbMaxPrimitivePerBox, const std::string& filename  );
   ~ObjScene(void);

protected:
   virtual void doInitialize();
   virtual void doPostInitialize();
   virtual void doAnimate();
   virtual void doAddLights();

private:
   std::vector<std::string> m_objFiles;
   std::string m_filename;
   Vertex m_objectScale;
   Vertex m_actorPosition;
};

