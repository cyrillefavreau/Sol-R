/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include "Scene.h"

struct IFTFaceTracker;

class KinectFaceTrackingScene :
        public Scene
{

public:
    KinectFaceTrackingScene(const std::string& name, const int nbMaxPrimitivePerBox);
    ~KinectFaceTrackingScene(void);

protected:
    virtual void doInitialize();
    virtual void doAnimate();
    virtual void doAddLights();

protected:
    IFTFaceTracker* m_faceTracker;
};
