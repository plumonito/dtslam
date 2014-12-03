#ifdef GL_ES_VERSION_2_0
precision mediump float;
#endif

uniform sampler2D uTexture;
uniform float uAlpha;
varying vec2 vTexCoord;

void main(void)
{
    vec4 texColor = texture2D(uTexture, vTexCoord);
	gl_FragColor = vec4(texColor.r, texColor.g, texColor.b, uAlpha);
}
