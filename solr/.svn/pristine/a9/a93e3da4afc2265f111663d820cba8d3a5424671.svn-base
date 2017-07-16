/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#include <fstream>
#include <vector>
#include <map>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "Consts.h"
#include "Logging.h"
#include "PDBReader.h"

//#define CONNECTIONS

struct Element
{
	std::string name;
	float radius;
	int materialId;
};

struct Atom
{
	int    processed;
	int    id;
	int    index;
	float4 position;
	int    materialId;
	int    chainId;
	int    residue;
	bool   isBackbone;
	bool   isWater;
};

struct Connection
{
	int atom1;
	int atom2;
};

const int NB_ELEMENTS = 70;
const float DEFAULT_ATOM_DISTANCE = 30.f;
const float DEFAULT_STICK_DISTANCE = 1.7f;
const Element elements[NB_ELEMENTS] =
{
	{ "C"  , 67.f,  1 },
	{ "N"  , 56.f,  2 },
	{ "O"  , 48.f,  3 },
	{ "H"  , 53.f,  4 },
	{ "B"  , 87.f,  5 },
	{ "F"  , 42.f,  6 },
	{ "P"  , 98.f,  7 },
	{ "S"  , 88.f,  8 },
	{ "V"  ,171.f,  9 },
	{ "K"  ,243.f, 10 },
	{ "HE" , 31.f, 11 },
	{ "LI" ,167.f, 12 },
	{ "BE" ,112.f, 13 },
	{ "NE" , 38.f, 14 },
	{ "NA" ,190.f, 15 },
	{ "MG" ,145.f, 16 },
	{ "AL" ,118.f, 17 },
	{ "SI" ,111.f, 18 },
	{ "CL" , 79.f, 19 },
	{ "AR" , 71.f, 20 },
	{ "CA" ,194.f, 21 },
	{ "SC" ,184.f, 22 },
	{ "TI" ,176.f, 23 },
	{ "CR" ,166.f, 24 },
	{ "MN" ,161.f, 25 },
	{ "FE" ,156.f, 26 },
	{ "CO" ,152.f, 27 },
	{ "NI" ,149.f, 28 },
	{ "CU" ,145.f, 29 },
	{ "ZN" ,142.f, 30 },
	{ "GA" ,136.f, 31 },
	{ "GE" ,125.f, 32 },
	{ "AS" ,114.f, 33 },
	{ "SE" ,103.f, 34 },
	{ "BR" , 94.f, 35 },
	{ "KR" , 88.f, 36 }, 

	// TODO
	{ "OD1" , 50.f, 37 },
	{ "OD2" , 50.f, 38 },
	{ "CG1" , 50.f, 39 }, 
	{ "CG2" , 50.f, 40 },
	{ "CD1" , 50.f, 41 },
	{ "CB"  , 50.f, 42 },
	{ "CG"  , 50.f, 43 },
	{ "CD"  , 50.f, 44 },
	{ "OE1" , 50.f, 45 },
	{ "NE2" , 50.f, 46 },
	{ "CZ"  , 50.f, 47 },
	{ "NH1" , 50.f, 48 },
	{ "NH2" , 50.f, 49 },
	{ "CD2" , 50.f, 50 },
	{ "CE1" , 50.f, 51 },
	{ "CE2" , 50.f, 52 },
	{ "CE"  , 50.f, 53 },
	{ "NZ"  , 50.f, 54 },
	{ "OH"  , 50.f, 55 },
	{ "CE"  , 50.f, 56 },
	{ "ND1" , 50.f, 57 },
	{ "ND2" , 50.f, 58 },
	{ "OXT" , 50.f, 59 },
	{ "OG1" , 50.f, 60 },
	{ "NE1" , 50.f, 61 },
	{ "CE3" , 50.f, 62 },
	{ "CZ2" , 50.f, 63 },
	{ "CZ3" , 50.f, 64 },
	{ "CH2" , 50.f, 65 },
	{ "OE2" , 50.f, 66 },
	{ "OG"  , 50.f, 67 },
	{ "OE2" , 50.f, 68 },
	{ "SD"  , 50.f, 69 },
	{ "SG"  , 50.f, 70 }
};

PDBReader::PDBReader(void) : m_nbBoxes(0), m_nbPrimitives(0)
{
}

PDBReader::~PDBReader(void)
{
}

Vertex PDBReader::loadAtomsFromFile(
	const std::string& filename,
	GPUKernel&   cudaKernel,
	GeometryType geometryType,
	const float  defaultAtomSize,
	const float  defaultStickSize,
	const int    materialType,
	const Vertex scale,
	const bool   useModels)
{
	int frame(0);
	int chainSelection(rand()%2);
	cudaKernel.resetBoxes(true);

	float distanceRatio = 2.f; //(geometryType==gtSticks || geometryType==gtAtomsAndSticks) ? 2.f : 1.f;

	std::map<int,Atom> atoms;
	std::vector<Connection> connections;
	float4 minPos = {  100000.f,  100000.f,  100000.f, 0.f };
	float4 maxPos = { -100000.f, -100000.f, -100000.f, 0.f };

   LOG_INFO(1,"--------------------------------------------------------------------------------" );
   LOG_INFO(1,"Loading PDB File: " << filename );
	int index(0);
	std::ifstream file(filename.c_str());
	if( file.is_open() )
	{
		while( file.good() )
		{
			std::string line;
			std::string value;
			std::getline( file, line );
			if( line.find("ATOM") == 0 /* || line.find("HETATM") == 0 */ )
			{
				// Atom
				Atom atom;
				atom.index = index;
				index++;
				std::string atomName;
				std::string chainId;
				std::string atomCode;
				size_t i(0);
				while( i<line.length() )
				{
					switch(i)
					{
					case 6: //ID
					case 12:
					case 76: // Atom name
					case 22: // ChainID
					case 30: // x
					case 38: // y
					case 46: // z
						value = "";
						break;
					case 21: atom.chainId = (int)line.at(i)-64; break;
					case 11: atom.id = static_cast<int>(atoi(value.c_str())); break;
					case 17: atomCode = value; break;
					case 79: atomName = value; break;
					case 26: 
						atom.residue = static_cast<int>(atoi(value.c_str())); 
						break;
					case 37: atom.position.x = static_cast<float>(atof(value.c_str())); break;
					case 45: atom.position.y = static_cast<float>(atof(value.c_str())); break;
					case 53: atom.position.z = -static_cast<float>(atof(value.c_str())); break;
					default:
						if( line.at(i) != ' ' ) value += line.at(i);
						break;
					}
					i++;
				}

				// Backbone
            atom.isBackbone = (geometryType==gtBackbone || geometryType==gtIsoSurface || atomCode.length()==1);

				// Material
				atom.materialId = 1;
				i=0;
				bool found(false);
				while( !found && i<NB_ELEMENTS )
				{
					if( atomName == elements[i].name )
					{
						found = true;
						switch( materialType )
						{
						case 1: 
							atom.materialId = (atom.chainId%2==0) ? elements[i].materialId%10 : 10; 
							break;
						case 2: 
							atom.materialId = atom.residue%10; 
							break;
						default: 
							atom.materialId = elements[i].materialId%10; 
							break;
						}
						atom.position.w = (geometryType==gtFixedSizeAtoms) ? defaultAtomSize : elements[i].radius;
					}
					++i;
				}

				if( geometryType!=gtBackbone || atom.isBackbone ) 
				{
					// Compute molecule size
					// min
					minPos.x = (atom.position.x < minPos.x) ? atom.position.x : minPos.x;
					minPos.y = (atom.position.y < minPos.y) ? atom.position.y : minPos.y;
					minPos.z = (atom.position.z < minPos.z) ? atom.position.z : minPos.z;

					// max
					maxPos.x = (atom.position.x > maxPos.x) ? atom.position.x : maxPos.x;
					maxPos.y = (atom.position.y > maxPos.y) ? atom.position.y : maxPos.y;
					maxPos.z = (atom.position.z > maxPos.z) ? atom.position.z : maxPos.z;

					// add Atom to the list
					atom.processed = 0;
					if( geometryType==gtSticks || (geometryType==gtAtomsAndSticks && atom.residue%2 == 0) ) 
					{
						atoms[atom.id] = atom;
					}
					else
					{
						atoms[index] = atom;
					}
				}
			}
		}
		file.close();
	}

   LOG_INFO(1,"Number of elements: " << atoms.size());
   LOG_INFO(1,"Building internal structures");
	Vertex objectSize;
	objectSize.x = (maxPos.x-minPos.x);
	objectSize.y = (maxPos.y-minPos.y);
	objectSize.z = (maxPos.z-minPos.z);

	float4 center;
	center.x = (minPos.x+maxPos.x)/2.f;
	center.y = (minPos.y+maxPos.y)/2.f;
	center.z = (minPos.z+maxPos.z)/2.f;

   Vertex objectScale;
	objectScale.x = scale.x/( maxPos.x - minPos.x);
	objectScale.y = scale.y/( maxPos.y - minPos.y);
	objectScale.z = scale.z/( maxPos.z - minPos.z);

   float atomDistance(DEFAULT_ATOM_DISTANCE);

	std::map<int,Atom>::iterator it = atoms.begin();
	while( it != atoms.end() )
	{
		Atom& atom((*it).second);
		if( atom.processed<2 )
		{
			int nb;

         float radius(atom.position.w);
			float stickRadius(atom.position.w);
         switch( geometryType )
         {
            case gtFixedSizeAtoms:
               radius = defaultAtomSize;
               break;
            case gtSticks:
               radius      = defaultStickSize;
               stickRadius = defaultStickSize;
               break;
            case gtAtomsAndSticks:
               radius      = atom.position.w/2.f;
               stickRadius = defaultStickSize/2.f;
               break;
            case gtBackbone:
               radius      = defaultStickSize;
               stickRadius = defaultStickSize;
               break;
            case gtIsoSurface:
               radius      = atom.position.w;
               break;
         }

         if( geometryType==gtSticks || geometryType==gtAtomsAndSticks || geometryType==gtBackbone ) 
			{
				std::map<int,Atom>::iterator it2 = atoms.begin();
				while( it2 != atoms.end() )
				{

					if( it2 != it && (*it2).second.processed<2 && ((*it).second.isBackbone==(*it2).second.isBackbone))
					{
						Atom& atom2((*it2).second);
						float4 a;
						a.x = atom.position.x - atom2.position.x;
						a.y = atom.position.y - atom2.position.y;
						a.z = atom.position.z - atom2.position.z;
						float distance = sqrtf( a.x*a.x + a.y*a.y + a.z*a.z );
						float stickDistance = (geometryType==gtBackbone && atom2.isBackbone) ? DEFAULT_STICK_DISTANCE*2.f : DEFAULT_STICK_DISTANCE;
						
                  if( distance < stickDistance )
						{
							float4 halfCenter;
							halfCenter.x = (atom.position.x + atom2.position.x)/2.f;
							halfCenter.y = (atom.position.y + atom2.position.y)/2.f;
							halfCenter.z = (atom.position.z + atom2.position.z)/2.f;

                     // Sticks
                     nb = cudaKernel.addPrimitive(ptCylinder,true);
							cudaKernel.setPrimitive( 
								nb,
								objectScale.x*distanceRatio*atomDistance*(atom.position.x - center.x), 
								objectScale.y*distanceRatio*atomDistance*(atom.position.y - center.y), 
								objectScale.z*distanceRatio*atomDistance*(atom.position.z - center.z), 
								objectScale.x*distanceRatio*atomDistance*(halfCenter.x - center.x), 
								objectScale.y*distanceRatio*atomDistance*(halfCenter.y - center.y), 
								objectScale.z*distanceRatio*atomDistance*(halfCenter.z - center.z),
								objectScale.x*stickRadius, 0.f,0.f,
                        (geometryType==gtSticks) ? atom.materialId : 11 );
                     Vertex vt0={0.f,0.f,0.f};
                     Vertex vt1={1.f,1.f,0.f};
                     Vertex vt2={0.f,0.f,0.f};
							cudaKernel.setPrimitiveTextureCoordinates( 
								nb, vt0, vt1, vt2 );
						}
					}
					it2++;
				}
			}

         bool addAtom( true );

         int m = atom.materialId;
			if( useModels && (atom.chainId%2==chainSelection) )
			{
            //addAtom = false;
            radius = stickRadius;
            if( geometryType==gtAtomsAndSticks)  m = 11;
			}

			if( addAtom )
			{
            // Enveloppe
            Vertex vt0={0.f,0.f,0.f};
            Vertex vt1={1.f,1.f,0.f};
            Vertex vt2={0.f,0.f,0.f};
            if( geometryType==gtIsoSurface && atom.isBackbone && atom.chainId%2==0 )
            {
					nb = cudaKernel.addPrimitive(ptSphere,true);
					cudaKernel.setPrimitive( 
						nb,
					   objectScale.x*distanceRatio*atomDistance*(atom.position.x - center.x), 
					   objectScale.y*distanceRatio*atomDistance*(atom.position.y - center.y), 
					   objectScale.z*distanceRatio*atomDistance*(atom.position.z - center.z), 
					   objectScale.x*radius*2.f, 0.f, 0.f,
						10 );
					cudaKernel.setPrimitiveTextureCoordinates( 
						nb, vt0, vt1, vt2 );
            }

				nb = cudaKernel.addPrimitive(ptSphere,true);
				cudaKernel.setPrimitive( 
					nb, 
					objectScale.x*distanceRatio*atomDistance*(atom.position.x - center.x), 
					objectScale.y*distanceRatio*atomDistance*(atom.position.y - center.y), 
					objectScale.z*distanceRatio*atomDistance*(atom.position.z - center.z), 
					objectScale.x*radius, 0.f, 0.f,
               m );
				cudaKernel.setPrimitiveTextureCoordinates( 
					nb, vt0, vt1, vt2 );
			}
		}
		++it;
	}
	objectSize.x *= objectScale.x*distanceRatio*atomDistance;
	objectSize.y *= objectScale.y*distanceRatio*atomDistance;
	objectSize.z *= objectScale.z*distanceRatio*atomDistance;
   LOG_INFO(1, "Loaded " << filename << " into frame " << cudaKernel.getFrame() << " [" << cudaKernel.getNbActivePrimitives() << " primitives]" );
   LOG_INFO(1, "Object size after scaling: " << objectSize.x << "," << objectSize.y << "," << objectSize.z );
   LOG_INFO(1,"--------------------------------------------------------------------------------" );
	return objectSize;
}
