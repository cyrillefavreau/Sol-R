/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include "Scene.h"

class XmasScene : public Scene
{

public:
    XmasScene( const std::string& name, const int nbMaxPrimitivePerBox  );
    ~XmasScene(void);

protected:
    virtual void doInitialize();
    virtual void doAnimate();
    virtual void doAddLights();

private:
    void createTree( int iteration, int boxId, int maxIterations, Vertex center, int material, float interval, float radius );

};
