#ifdef GL_ES_VERSION_2_0
precision mediump float;
#endif

uniform sampler2D uTexture;
varying vec2 vTexCoord;

void main(void)
{
    gl_FragColor = texture2D(uTexture, vTexCoord);
}
