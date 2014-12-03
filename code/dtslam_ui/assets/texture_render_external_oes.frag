#extension GL_OES_EGL_image_external : require

#ifdef GL_ES_VERSION_2_0
precision mediump float;
#endif

uniform samplerExternalOES uTexture;
varying vec2 vTexCoord;

void main(void)
{
    gl_FragColor = texture2D(uTexture, vTexCoord);
}
