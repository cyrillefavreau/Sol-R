/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include "Scene.h"

class ColumnScene :
   public Scene
{

public:
   ColumnScene( const std::string& name, const int nbMaxPrimitivePerBox  );
   ~ColumnScene(void);

protected:
   virtual void doInitialize();
   virtual void doAnimate();
   virtual void doAddLights();

private:
   Vertex m_actorPosition;
   Vertex m_objectScale;

};

