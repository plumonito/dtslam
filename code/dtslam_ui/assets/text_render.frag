#ifdef GL_ES_VERSION_2_0
precision mediump float;
#endif

uniform sampler2D uTexture;
uniform vec4 uColor;
varying vec2 vTexCoord;

void main(void)
{
    vec4 texColor = texture2D(uTexture, vTexCoord);
	gl_FragColor = vec4(uColor.r, uColor.g, uColor.b, uColor.a*texColor.a);
}
