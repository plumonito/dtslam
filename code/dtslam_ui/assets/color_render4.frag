#ifdef GL_ES_VERSION_2_0
precision mediump float;
#endif

uniform sampler2D uTexture;
varying vec4 vColor;

void main(void)
{
    gl_FragColor = vColor;
}
