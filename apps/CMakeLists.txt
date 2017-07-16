ADD_EXECUTABLE(
  solrViewer
  solrViewer.cpp
  scenes/SpheresScene.cpp
  scenes/Scene.cpp
  scenes/SpindizzyScene.cpp
  scenes/CornellBoxScene.cpp
  scenes/FractalsScene.cpp
  scenes/FractalScene.cpp
  scenes/ObjScene.cpp
  scenes/Year2013.cpp
  scenes/CausticScene.cpp
  scenes/CubesScene.cpp
  scenes/TrefoilKnotScene.cpp
  scenes/XmasScene.cpp
  scenes/KinectFaceTrackingScene.cpp
  scenes/CylinderScene.cpp
  scenes/MoleculeScene.cpp
  scenes/PerpetualMotionScene.cpp
  scenes/GraphScene.cpp
  scenes/TransparentScene.cpp
  scenes/AnimationScene.cpp
  scenes/WaterScene.cpp
  scenes/MetaballsScene.cpp
  scenes/TrianglesScene.cpp
  scenes/DoggyStyleScene.cpp
  scenes/GalleryScene.cpp
  scenes/SwcScene.cpp
)

include_directories(../solr)

SET(
  SOLRVIEWER_HEADERS
  scenes/SpheresScene.h
  scenes/Scene.h
  scenes/SpindizzyScene.h
  scenes/CornellBoxScene.h
  scenes/FractalsScene.h
  scenes/FractalScene.h
  scenes/ObjScene.h
  scenes/Year2013.h
  scenes/CausticScene.h
  scenes/CubesScene.h
  scenes/TrefoilKnotScene.h
  scenes/XmasScene.h
  scenes/KinectFaceTrackingScene.h
  scenes/CylinderScene.h
  scenes/MoleculeScene.h
  scenes/PerpetualMotionScene.h
  scenes/GraphScene.h
  scenes/TransparentScene.h
  scenes/AnimationScene.h
  scenes/WaterScene.h
  scenes/MetaballsScene.h
  scenes/TrianglesScene.h
  scenes/DoggyStyleScene.h
  scenes/GalleryScene.h
  scenes/SwcScene.h
)

TARGET_LINK_LIBRARIES(
    solrViewer
    solr
    ${OPENCL_LIBRARIES}
    ${GLUT_LIBRARIES}
    )

# ------------------------------------------------------------
INSTALL(TARGETS solrViewer DESTINATION bin)
# ------------------------------------------------------------

