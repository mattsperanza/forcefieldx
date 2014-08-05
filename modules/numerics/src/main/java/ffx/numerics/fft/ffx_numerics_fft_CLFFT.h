/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class ffx_numerics_fft_CLFFT */

#ifndef _Included_ffx_numerics_fft_CLFFT
#define _Included_ffx_numerics_fft_CLFFT
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     ffx_numerics_fft_CLFFT
 * Method:    clfftSetupNative
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_ffx_numerics_fft_CLFFT_clfftSetupNative
  (JNIEnv *, jclass);

/*
 * Class:     ffx_numerics_fft_CLFFT
 * Method:    clfftCreateDefaultPlanNative
 * Signature: (JIIII)J
 */
JNIEXPORT jlong JNICALL Java_ffx_numerics_fft_CLFFT_clfftCreateDefaultPlanNative
  (JNIEnv *, jclass, jlong, jint, jint, jint, jint);

/*
 * Class:     ffx_numerics_fft_CLFFT
 * Method:    clfftSetPlanPrecisionNative
 * Signature: (JI)I
 */
JNIEXPORT jint JNICALL Java_ffx_numerics_fft_CLFFT_clfftSetPlanPrecisionNative
  (JNIEnv *, jclass, jlong, jint);

/*
 * Class:     ffx_numerics_fft_CLFFT
 * Method:    clfftSetLayoutNative
 * Signature: (JII)I
 */
JNIEXPORT jint JNICALL Java_ffx_numerics_fft_CLFFT_clfftSetLayoutNative
  (JNIEnv *, jclass, jlong, jint, jint);

/*
 * Class:     ffx_numerics_fft_CLFFT
 * Method:    clfftExecuteTransformNative
 * Signature: (JIJJJ)I
 */
JNIEXPORT jint JNICALL Java_ffx_numerics_fft_CLFFT_clfftExecuteTransformNative
  (JNIEnv *, jclass, jlong, jint, jlong, jlong, jlong);

/*
 * Class:     ffx_numerics_fft_CLFFT
 * Method:    clfftDestroyPlanNative
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_ffx_numerics_fft_CLFFT_clfftDestroyPlanNative
  (JNIEnv *, jclass, jlong);

#ifdef __cplusplus
}
#endif
#endif
