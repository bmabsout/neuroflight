#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <platform.h>
#include "build/build_config.h"
#include "build/debug.h"
#include "common/axis.h"
#include "common/maths.h"
#include "drivers/time.h"
#include "fc/fc_core.h"
#include "fc/fc_rc.h"
#include "fc/rc_controls.h"
#include "fc/runtime_config.h"
#include "neuroflight/neuro.h"
#include "flight/imu.h"
#include "flight/mixer.h"
#include "sensors/gyro.h"
#include "neuroflight/graph_interface.h"
#include "graph_dim.h"
#include "common/filter.h"
#include "io/serial.h"
#include "io/uart4Serial.h"
#include "trajectory_buffer.h"
// #include "tflite/model_data.h"
// #include "tflite/smooth_frozen2.h"
// #include "tflite/model_test_data.h"
// #include "tflite/large_test.h"
// #include "tflite/small_test.h"
// #include "tflite/small_test_no_mul_add.h"

/* An array containing inputs for the neural network 
 * where the first element is the oldest
 */
static float graphInput[GRAPH_INPUT_SIZE];
static float graphOutput[GRAPH_OUTPUT_SIZE];
static float controlOutput[GRAPH_OUTPUT_SIZE];
static float previousOutput[GRAPH_OUTPUT_SIZE];

static bool initFlag = true;



static FAST_RAM filterApplyFnPtr dtermNotchFilterApplyFn;
static FAST_RAM void *dtermFilterNotch[3];
static FAST_RAM filterApplyFnPtr dtermLpfApplyFn;
static FAST_RAM void *dtermFilterLpf[3];

static FAST_RAM float dT;




typedef union dtermFilterLpf_u {
    pt1Filter_t pt1Filter[3];
    biquadFilter_t biquadFilter[3];
    firFilterDenoise_t denoisingFilter[3];
} dtermFilterLpf_t;

void neuroInitFilters(const pidProfile_t *pidProfile)
{
    // BUILD_BUG_ON(FD_YAW != 2); // only setting up Dterm filters on roll and pitch axes, so ensure yaw axis is 2
    float targetLooptime = gyro.targetLooptime;
    dT = (float)gyro.targetLooptime * 0.000001f;
    if (targetPidLooptime == 0) {
        // no looptime set, so set all the filters to null
        dtermNotchFilterApplyFn = nullFilterApply;
        dtermLpfApplyFn = nullFilterApply;
        return;
    }

    const uint32_t pidFrequencyNyquist = (1.0f / dT) / 2; // No rounding needed

    uint16_t dTermNotchHz;
    if (pidProfile->dterm_notch_hz <= pidFrequencyNyquist) {
        dTermNotchHz = pidProfile->dterm_notch_hz;
    } else {
        if (pidProfile->dterm_notch_cutoff < pidFrequencyNyquist) {
            dTermNotchHz = pidFrequencyNyquist;
        } else {
            dTermNotchHz = 0;
        }
    }

    if (dTermNotchHz != 0 && pidProfile->dterm_notch_cutoff != 0) {
        static biquadFilter_t biquadFilterNotch[3];
        dtermNotchFilterApplyFn = (filterApplyFnPtr)biquadFilterApply;
        const float notchQ = filterGetNotchQ(dTermNotchHz, pidProfile->dterm_notch_cutoff);
        for (int axis = FD_ROLL; axis <= FD_YAW; axis++) {
            dtermFilterNotch[axis] = &biquadFilterNotch[axis];
            biquadFilterInit(dtermFilterNotch[axis], dTermNotchHz, targetPidLooptime, notchQ, FILTER_NOTCH);
        }
    } else {
        dtermNotchFilterApplyFn = nullFilterApply;
    }

    static dtermFilterLpf_t dtermFilterLpfUnion;
    if (pidProfile->dterm_lpf_hz == 0 || pidProfile->dterm_lpf_hz > pidFrequencyNyquist) {
        dtermLpfApplyFn = nullFilterApply;
    } else {
        switch (pidProfile->dterm_filter_type) {
        default:
            dtermLpfApplyFn = nullFilterApply;
            break;
        case FILTER_PT1:
            dtermLpfApplyFn = (filterApplyFnPtr)pt1FilterApply;
            for (int axis = FD_ROLL; axis <= FD_YAW; axis++) {
                dtermFilterLpf[axis] = &dtermFilterLpfUnion.pt1Filter[axis];
                pt1FilterInit(dtermFilterLpf[axis], pidProfile->dterm_lpf_hz, dT);
            }
            break;
        case FILTER_BIQUAD:
            dtermLpfApplyFn = (filterApplyFnPtr)biquadFilterApply;
            for (int axis = FD_ROLL; axis <= FD_YAW; axis++) {
                dtermFilterLpf[axis] = &dtermFilterLpfUnion.biquadFilter[axis];
                biquadFilterInitLPF(dtermFilterLpf[axis], pidProfile->dterm_lpf_hz, targetPidLooptime);
            }
            break;
        case FILTER_FIR:
            dtermLpfApplyFn = (filterApplyFnPtr)firFilterDenoiseUpdate;
            for (int axis = FD_ROLL; axis <= FD_YAW; axis++) {
                dtermFilterLpf[axis] = &dtermFilterLpfUnion.denoisingFilter[axis];
                firFilterDenoiseInit(dtermFilterLpf[axis], pidProfile->dterm_lpf_hz, targetPidLooptime);
            }
            break;
        }
    }
}

void neuroInit(const pidProfile_t *pidProfile)
{
    neuroInitFilters(pidProfile);
    for (int i = 0; i < GRAPH_OUTPUT_SIZE; i++) {
        previousOutput[i] = -1; 
    }

}

void evaluateGraphWithErrorStateDeltaStateAct(timeUs_t currentTimeUs){
    static timeUs_t previousTime;
    static float previousState[3];
    const float deltaT = ((float)(currentTimeUs - previousTime))/1000000.0f;
    previousTime = currentTimeUs;

    //Prepare the neural network inputs
    // Set the current error and deriviate
    for (int axis = FD_ROLL; axis <= FD_YAW; axis++) {

        float currentSetpoint = getSetpointRate(axis);
        float state = gyro.gyroADCf[axis];
        const float deltaState = state - previousState[axis]; 
        float error = currentSetpoint - state; 
        graphInput[axis] = error;

        if (debugMode == DEBUG_NN_GYDELTA) {
            debug[axis] = (int16_t)(1000*deltaState);
        }
        
        if (debugMode == DEBUG_NN_SP) {
            debug[axis] = (int16_t)(1000*currentSetpoint);
        }

        if (debugMode == DEBUG_NN_GYRATE) {
            debug[axis] = (int16_t)(10*state);
        }

        if (debugMode == DEBUG_NN_ERR_RATE) {
            debug[axis] = (int16_t)(10*error);
        }

        graphInput[axis + 3] = state;
        //TODO We need to include delta time because the loop is not fixed
        graphInput[axis + 6] = deltaState;

        previousState[axis] = state;
    }

    for (int i = 0; i < GRAPH_OUTPUT_SIZE; i++) {
        graphInput[i+9] = previousOutput[i];
    }
    
    // if (debugMode == DEBUG_NN_OUT) {
    //     for (int i = 0; i<GRAPH_INPUT_SIZE; i++){
    //         debug[i] = (int16_t)(graphInput[i] * 1000.0);
    //     }
    // }

    if (debugMode == DEBUG_NN_ACT_IN) {
        for (int i = 0; i<GRAPH_OUTPUT_SIZE; i++){
            debug[i] = (int16_t)(previousOutput[i] * 1000.0);
        }
    }

    rpy_t error;
    error.roll = 0.1;
    error.pitch = 0.2;
    error.yaw = 0.3;

    rpy_t ang_vel;
    ang_vel.roll = gyro.gyroADCf[0];
    ang_vel.pitch = gyro.gyroADCf[1];
    ang_vel.yaw = gyro.gyroADCf[2];

    rpy_t ang_acc;
    ang_acc.roll = 0.4;
    ang_acc.pitch = 0.5;
    ang_acc.yaw = 0.6;
    action_t prev_action;
    prev_action.top_left = 0;
    prev_action.top_right = 0;
    prev_action.bottom_left = 0;
    prev_action.bottom_right = 0;

    observation_t obs;
    obs.error = error;
    obs.ang_vel = ang_vel;
    obs.ang_acc = ang_acc;
    obs.prev_action = prev_action;

    // traj_transmission_handler(obs);
    
    serialWrite(uart4Serial, 'a');

    //Evaluate the neural network graph and convert to range [-1,1]->[0,1]
    // infer(graphInput, GRAPH_INPUT_SIZE, graphOutput, model_tflite, GRAPH_OUTPUT_SIZE);

    for (int i = 0; i < GRAPH_OUTPUT_SIZE; i++) {
        float new_output = graphOutput[i];
        new_output = constrainf(new_output, -1.0f, 1.0f);
        controlOutput[i] = transformScale(new_output, -1.0f, 1.0f, 0.0f, 1.0f);
        previousOutput[i] = new_output;
    }

    if (debugMode == DEBUG_NN_OUT) {
        for (int i = 0; i<GRAPH_OUTPUT_SIZE; i++){
            debug[i] = (int16_t)(previousOutput[i] * 1000.0);
        }
    }
}

#define MAX_BUFFER_SIZE 1000

#define buffer_size_t uint16_t
#define SIZE_BYTES sizeof(buffer_size_t)/sizeof(uint8_t)
#define crc_t uint16_t
#define CRC_BYTES sizeof(crc_t)/sizeof(uint8_t)
#define META_BYTES (SIZE_BYTES + CRC_BYTES)

uint8_t buffer[META_BYTES + MAX_BUFFER_SIZE];
buffer_size_t buffer_size = 0;

buffer_size_t expected_block_size() {
    if(SIZE_BYTES > buffer_size)
        return 0;
    buffer_size_t num_bytes = 0;
    for(int i=0; i < SIZE_BYTES; i++)
        num_bytes += ((buffer_size_t)buffer[i]) << (i*8);
    return num_bytes;
}

crc_t expected_crc() {
    buffer_size_t crc = 0;
    for(int i=SIZE_BYTES; i < SIZE_BYTES+CRC_BYTES; i++)
        crc += ((buffer_size_t)buffer[i]) << i*8;
    return crc;
}

buffer_size_t block_size() {
    return (buffer_size > META_BYTES) ? (buffer_size - META_BYTES) : 0;
}

void add_to_buffer(uint8_t add_me) {
    buffer[buffer_size] = add_me;
    buffer_size++;
}

uint8_t block_at(buffer_size_t i) {
    return buffer[i+META_BYTES];
}

void print_block() {
    for(int i = 0; i < block_size(); i++) {
        serialWrite(uart4Serial, block_at(i));
    }
    serialWrite(uart4Serial, '\n');
}

void update_nn() {
    // for(int i = 0; i < block_size(); i++) {
    //     model_tflite[i] = block_at(i);
    // }
}

crc_t block_crc() {
    uint8_t tmp;
    crc_t crcAccum = 0xffff;
    for (int i = 0; i < block_size(); i++) {
        tmp = block_at(i) ^ (uint8_t)(crcAccum & 0xff);
        tmp ^= (tmp << 4);
        crcAccum = (crcAccum >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4);
    }
    return crcAccum;
}


bool test = false;

void neuroController(timeUs_t currentTimeUs, const pidProfile_t *pidProfile){


    if(initFlag) {
        neuroInit(pidProfile);
        initUART4();
        bool *n = NULL;
        test = *n;
        initFlag = false;
    } else {
        uint8_t bytesWaiting;
        while ((bytesWaiting = serialRxBytesWaiting(uart4Serial))) {
            uint8_t b = serialRead(uart4Serial);
            add_to_buffer(b);
            if((buffer_size >= META_BYTES) && (block_size() == expected_block_size())) {
                // print_block();
                update_nn();
                serialWrite(uart4Serial, '0'+(block_crc() == expected_crc()));
                serialWrite(uart4Serial, '\n');
                buffer_size = 0;
            }
        };

        evaluateGraphWithErrorStateDeltaStateAct(currentTimeUs);
        mixGraphOutput(currentTimeUs, controlOutput);
    }
}

float transformScale(float value, float oldLow, float oldHigh, float newLow, float newHigh){
	return ((value - oldLow) / (oldHigh - oldLow)) * (newHigh - newLow) + newLow;
}
