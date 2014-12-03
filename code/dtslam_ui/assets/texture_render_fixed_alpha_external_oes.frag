#extension GL_OES_EGL_image_external : require
#ifdef GL_ES_VERSION_2_0
precision mediump float;
#endif

uniform samplerExternalOES uTexture;
uniform float uAlpha;
varying vec2 vTexCoord;

void main(void)
{
    vec4 texColor = texture2D(uTexture, vTexCoord);
	gl_FragColor = vec4(texColor.r, texColor.g, texColor.b, uAlpha);
}
