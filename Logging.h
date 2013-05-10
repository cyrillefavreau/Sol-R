#pragma once

#ifdef WIN32
   #define _CRT_SECURE_NO_WARNINGS
	#ifdef LOGGING
		#include <ETWLoggingModule.h>
		#include <ETWResources.h>
	#else
		#include <sstream>
	#endif
	#include <windows.h>
#endif // WIN32

#include <ctime>

inline std::string getTimestamp()
{
   time_t rawtime;
   struct tm * timeinfo;
   char buffer [80];

   time ( &rawtime );
   timeinfo = localtime( &rawtime );

   strftime (buffer,80,"%Y-%m-%d-%H-%M-%S",timeinfo);
   return std::string(buffer);
}

#ifdef WIN32
#if 1
	#define LOG_INFO( __level, __msg ) \
   { \
      if( __level==1) std::cout << getTimestamp() << " INFO  [" << __level << "] " << __msg << std::endl; \
   }
	#define LOG_ERROR( __msg ) \
      std::cout << getTimestamp() << " ERROR [1] " << __msg << std::endl;
#else
	#define LOG_INFO( __level, __msg ) \
	   { \
		  if( m_activeLogging ) \
		  { \
			 std::stringstream __s; \
			 __s << "[INFO]  " << __msg; \
			 OutputDebugString(__s.str().c_str()); \
		  } \
	   }

	#define LOG_ERROR( __msg ) \
	   { \
		  std::stringstream __s; \
		  __s << "[ERROR] " << __msg; \
		  OutputDebugString(__s.str().c_str()); \
	   }
#endif // 0
#else
	#define LOG_INFO( __level, __msg )
	#define LOG_ERROR( __msg )
#endif
