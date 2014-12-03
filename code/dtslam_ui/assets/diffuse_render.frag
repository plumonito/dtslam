#ifdef GL_ES_VERSION_2_0
precision mediump float;
#endif

uniform sampler2D uTexture;
varying vec4 vColor;
varying vec3 vNormal;
varying vec4 vVertexPos;

void main(void)
{
	vec4 lightPos = vec4(0.0,0.0,0.0,1.0);
	vec4 lightDir4 = vVertexPos - lightPos;
	vec3 lightDir = normalize(lightDir4.xyz);
	float lightIntensity=2.0;
	float diffuseWeight = max(0.0, lightIntensity*dot(lightDir,normalize(vNormal)));
	float weight = clamp(diffuseWeight + 0.4,0.0,1.0);
    //gl_FragColor = vec4(vColor.rgb*weight, vColor.a);
	
	gl_FragColor = vColor;
}
