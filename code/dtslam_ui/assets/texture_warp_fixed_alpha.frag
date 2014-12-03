#ifdef GL_ES_VERSION_2_0
precision mediump float;
#endif

uniform sampler2D uTexture;
uniform mat3 uHomography;
uniform float uAlpha;
varying vec2 vTexCoord;

void main(void)
{
	vec3 dstCoord3 = uHomography * vec3(vTexCoord,1.0);
	vec2 dstCoord2 = dstCoord3.xy / dstCoord3.z;
	//vec2 dstCoord2 = vTexCoord;
    vec4 texColor = texture2D(uTexture, dstCoord2);
	gl_FragColor = vec4(texColor.r, texColor.g, texColor.b, uAlpha);
	//gl_FragColor = texColor;
	//gl_FragColor = vec4(1.0,1.0,1.0,1.0);
}
