#ifdef GL_ES_VERSION_2_0
precision mediump float;
#endif

uniform sampler2D uTexture;
uniform mat3 uHomography;
varying vec2 vTexCoord;

void main(void)
{
	vec3 dstCoord3 = uHomography * vec3(vTexCoord,1.0);
	//vec2 dstCoord2 = dstCoord3.xy / dstCoord3.z;
	//vec2 dstCoord2 = vTexCoord;
#ifdef GL_ES_VERSION_2_0
    gl_FragColor = texture2D(uTexture, dstCoord3);
#else
    gl_FragColor = texture2D(uTexture, dstCoord3.xy, dstCoord3.z);
#endif
	//gl_FragColor = vec4(1.0,1.0,1.0,1.0);
}
