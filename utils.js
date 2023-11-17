function multiplyMatrices(matrixA, matrixB) {
    var result = [];

    for (var i = 0; i < 4; i++) {
        result[i] = [];
        for (var j = 0; j < 4; j++) {
            var sum = 0;
            for (var k = 0; k < 4; k++) {
                sum += matrixA[i * 4 + k] * matrixB[k * 4 + j];
            }
            result[i][j] = sum;
        }
    }

    // Flatten the result array
    return result.reduce((a, b) => a.concat(b), []);
}
function createIdentityMatrix() {
    return new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]);
}
function createScaleMatrix(scale_x, scale_y, scale_z) {
    return new Float32Array([
        scale_x, 0, 0, 0,
        0, scale_y, 0, 0,
        0, 0, scale_z, 0,
        0, 0, 0, 1
    ]);
}

function createTranslationMatrix(x_amount, y_amount, z_amount) {
    return new Float32Array([
        1, 0, 0, x_amount,
        0, 1, 0, y_amount,
        0, 0, 1, z_amount,
        0, 0, 0, 1
    ]);
}

function createRotationMatrix_Z(radian) {
    return new Float32Array([
        Math.cos(radian), -Math.sin(radian), 0, 0,
        Math.sin(radian), Math.cos(radian), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ])
}

function createRotationMatrix_X(radian) {
    return new Float32Array([
        1, 0, 0, 0,
        0, Math.cos(radian), -Math.sin(radian), 0,
        0, Math.sin(radian), Math.cos(radian), 0,
        0, 0, 0, 1
    ])
}

function createRotationMatrix_Y(radian) {
    return new Float32Array([
        Math.cos(radian), 0, Math.sin(radian), 0,
        0, 1, 0, 0,
        -Math.sin(radian), 0, Math.cos(radian), 0,
        0, 0, 0, 1
    ])
}

function getTransposeMatrix(matrix) {
    return new Float32Array([
        matrix[0], matrix[4], matrix[8], matrix[12],
        matrix[1], matrix[5], matrix[9], matrix[13],
        matrix[2], matrix[6], matrix[10], matrix[14],
        matrix[3], matrix[7], matrix[11], matrix[15]
    ]);
}

const vertexShaderSource = `
attribute vec3 position;
attribute vec3 normal; // Normal vector for lighting

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 normalMatrix;

uniform vec3 lightDirection;

varying vec3 vNormal;
varying vec3 vLightDirection;

void main() {
    vNormal = vec3(normalMatrix * vec4(normal, 0.0));
    vLightDirection = lightDirection;

    gl_Position = vec4(position, 1.0) * projectionMatrix * modelViewMatrix; 
}

`

const fragmentShaderSource = `
precision mediump float;

uniform vec3 ambientColor;
uniform vec3 diffuseColor;
uniform vec3 specularColor;
uniform float shininess;

varying vec3 vNormal;
varying vec3 vLightDirection;

void main() {
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(vLightDirection);
    
    // Ambient component
    vec3 ambient = ambientColor;

    // Diffuse component
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * diffuseColor;

    // Specular component (view-dependent)
    vec3 viewDir = vec3(0.0, 0.0, 1.0); // Assuming the view direction is along the z-axis
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = spec * specularColor;

    gl_FragColor = vec4(ambient + diffuse + specular, 1.0);
}

`

/**
 * @WARNING DO NOT CHANGE ANYTHING ABOVE THIS LINE
 */



/**
 * 
 * @TASK1 Calculate the model view matrix by using the chatGPT
 */

function getChatGPTModelViewMatrix() {
    // the output that gpt gave us
    const transformationMatrix = new Float32Array([
        0.3535533845424652, -0.3061862292289734, 0.6123724579811096, 0.3,
        0.3535533845424652, 0.9185581207275391, -0.1767767072916031, -0.25,
        -0.8660253882408142, -0.25, 0.4330126941204071, 0,
        0, 0, 0, 1
    ]);
    
    return getTransposeMatrix(transformationMatrix);
}


/**
 * 
 * @TASK2 Calculate the model view matrix by using the given 
 * transformation methods and required transformation parameters
 * stated in transformation-prompt.txt
 */
function getModelViewMatrix() {
    // calculate the model view matrix by using the transformation
    // methods and return the modelView matrix in this method
    let modelViewMatrix = createIdentityMatrix();

    // Apply translation
    //createTranslationMatrix creates 0.5 of of object a new matrix
    // multiply matrices function multiplies two matrixes and creates a 3D object
    //for example (3x1) x (1x3) = 3x3 matrix
    
    const translationMatrix = createTranslationMatrix(0.5, 0.5, 0.5);
    modelViewMatrix = multiplyMatrices(modelViewMatrix, translationMatrix);

    // Apply scaling (make the cube smaller) // Because I did it with 1.5 1.5 1.5 ıt was not fully visible in the screen
    // In task 2 I want to see the whole cube initially it was very very big
    // I just can able to see just a small part of cube
    const scalingMatrix = createScaleMatrix(0.5, 0.5, 0.5); // Adjust the scaling factor as needed
    modelViewMatrix = multiplyMatrices(modelViewMatrix, scalingMatrix);

    //Rotation 30 degrees
    const rotationMatrixZ = createRotationMatrix_Z((30 * Math.PI) / 180.0);
    modelViewMatrix = multiplyMatrices(modelViewMatrix, rotationMatrixZ);

    // Rotation 45 degrees
    const rotationMatrixX = createRotationMatrix_X((45 * Math.PI) / 180.0);
    modelViewMatrix = multiplyMatrices(modelViewMatrix, rotationMatrixX);

    // Rotation 90 degrees
    const rotationMatrixY = createRotationMatrix_Y((90 * Math.PI) / 180.0);
    modelViewMatrix = multiplyMatrices(modelViewMatrix, rotationMatrixY);

    //console.log(getModelViewMatrix)
    

    // Return the final model-view matrix
    return modelViewMatrix;
}

/**
 * 
 * @TASK3 Ask CHAT-GPT to animate the transformation calculated in 
 * task2 infinitely with a period of 10 seconds. 
 * First 5 seconds, the cube should transform from its initial 
 * position to the target position.
 * The next 5 seconds, the cube should return to its initial position.
 */
function getPeriodicMovement(startTime) {
    // this metdo should return the model view matrix at the given time
    // to get a smooth animation
    const elapsedTime = (Date.now() - startTime) % 10000; // 10-second interval

    if (elapsedTime < 5000) {
        // First 5 seconds: Transition to the calculated transformation in task 2
        const progress = elapsedTime / 5000; // Normalize the progress from 0 to 1
        const task2Matrix = getModelViewMatrix(); // Use the transformation from task 2
        const identityMatrix = createIdentityMatrix();
        return interpolateMatrices(identityMatrix, task2Matrix, progress);
    } else {
        // Last 5 seconds: Return to the initial position
        const progress = (elapsedTime - 5000) / 5000; // Normalize the progress from 0 to 1
        const task2Matrix = getModelViewMatrix(); // Use the transformation from task 2
        const identityMatrix = createIdentityMatrix();
        return interpolateMatrices(task2Matrix, identityMatrix, progress);
    }
}

function interpolateMatrices(matrixA, matrixB, progress) {
    // Linear interpolation between two matrices
    const result = [];
    for (let i = 0; i < matrixA.length; i++) {
        result[i] = matrixA[i] + progress * (matrixB[i] - matrixA[i]);
    }
    return result;
}

