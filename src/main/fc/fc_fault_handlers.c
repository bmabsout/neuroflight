/*
 * This file is part of Cleanflight.
 *
 * Cleanflight is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Cleanflight is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Cleanflight.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "platform.h"

#include "drivers/light_led.h"
#include "drivers/time.h"
#include "drivers/transponder_ir.h"

#include "fc/fc_init.h"

#include "io/serial.h"

#include "flight/mixer.h"

#if defined(STM32F7)
#include "stm32f7xx_hal.h"
#else
#error "configuration not supported yet!"
#endif

typedef struct PG_PACKED crashStackContext
{
    uint32_t r0;
    uint32_t r1;
    uint32_t r2;
    uint32_t r3;
    uint32_t r12;
    uint32_t lr;
    uint32_t return_address;
    uint32_t xpsr;
} crashStackContext_t;

#define JMP_INTO_SECONDARY_FAULT_HANDLER() __asm volatile(          \
    " tst lr, #4                                                \n" \
    " ite eq                                                    \n" \
    " mrseq r0, msp                                             \n" \
    " mrsne r0, psp                                             \n" \
    " ldr r1, [r0, #24]                                         \n" \
    " bl secondary_fault_handler                                 \n")

extern void SystemClock_Config(void);

void delay_dumb(int ms)
{
    for (int i = 0; i < ms * 200000; i++)
        asm volatile("\tnop\n");
}

void secondary_fault_handler(crashStackContext_t *ctxt)
{
    volatile unsigned long _CFSR;
    volatile unsigned long _HFSR;
    volatile unsigned long _DFSR;
    volatile unsigned long _AFSR;
    volatile unsigned long _BFAR;
    volatile unsigned long _MMAR;
    // Configurable Fault Status Register
    // Consists of MMSR, BFSR and UFSR
    _CFSR = (*((volatile unsigned long *)(0xE000ED28)));

    // Hard Fault Status Register
    _HFSR = (*((volatile unsigned long *)(0xE000ED2C)));

    // Debug Fault Status Register
    _DFSR = (*((volatile unsigned long *)(0xE000ED30)));

    // Auxiliary Fault Status Register
    _AFSR = (*((volatile unsigned long *)(0xE000ED3C)));

    // Read the Fault Address Registers. These may not contain valid values.
    // Check BFARVALID/MMARVALID to see if they are valid values
    // MemManage Fault Address Register
    _MMAR = (*((volatile unsigned long *)(0xE000ED34)));
    // Bus Fault Address Register
    _BFAR = (*((volatile unsigned long *)(0xE000ED38)));
    UART_HandleTypeDef huart;
    GPIO_InitTypeDef huart_pin_config;
    char buffer[128];
    size_t len = 0;
    LED2_OFF;
    LED1_OFF;
    LED0_ON; /*errored out*/
    delay_dumb(1000);

    huart_pin_config.Pin = GPIO_PIN_0 | GPIO_PIN_1;
    huart_pin_config.Mode = GPIO_MODE_AF_PP;
    huart_pin_config.Pull = GPIO_NOPULL;
    huart_pin_config.Speed = GPIO_SPEED_FREQ_LOW;
    huart_pin_config.Alternate = GPIO_AF7_USART3;
    HAL_GPIO_Init(GPIOA, &huart_pin_config);

    huart.Instance = USART3;
    huart.Init.BaudRate = 115200;
    huart.Init.WordLength = UART_WORDLENGTH_8B;
    huart.Init.StopBits = UART_STOPBITS_1;
    huart.Init.Parity = UART_PARITY_NONE;
    huart.Init.Mode = UART_MODE_TX_RX;
    huart.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart.Init.OverSampling = UART_OVERSAMPLING_16;
    huart.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
    huart.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
    HAL_UART_DeInit(&huart);
    if (HAL_UART_Init(&huart) != HAL_OK)
    {
        LED0_ON;
        while (1)
        {
            asm volatile("\tnop\n");
        }
    }
    while (1)
    {
        LED0_ON;
        delay_dumb(100);
        HAL_UART_Transmit(&huart, (uint8_t *)"-----------------------------------------------------\r\n", 56, 500);
        HAL_UART_Transmit(&huart, (uint8_t *)"fault handler\r\n", 19, 500);
        len = snprintf(buffer, 128, "lr: 0x%08x\r\n", ctxt->lr);
        HAL_UART_Transmit(&huart, (uint8_t *)buffer, len, 500);
        len = snprintf(buffer, 128, "r0: 0x%08x\r\n", ctxt->r0);
        HAL_UART_Transmit(&huart, (uint8_t *)buffer, len, 500);
        len = snprintf(buffer, 128, "r1: 0x%08x\r\n", ctxt->r1);
        HAL_UART_Transmit(&huart, (uint8_t *)buffer, len, 500);
        len = snprintf(buffer, 128, "r2: 0x%08x\r\n", ctxt->r2);
        HAL_UART_Transmit(&huart, (uint8_t *)buffer, len, 500);
        len = snprintf(buffer, 128, "r3: 0x%08x\r\n", ctxt->r3);
        HAL_UART_Transmit(&huart, (uint8_t *)buffer, len, 500);
        len = snprintf(buffer, 128, "r12: 0x%08x\r\n", ctxt->r12);
        HAL_UART_Transmit(&huart, (uint8_t *)buffer, len, 500);
        len = snprintf(buffer, 128, "ret addr: 0x%08x\r\n", ctxt->return_address);
        HAL_UART_Transmit(&huart, (uint8_t *)buffer, len, 500);
        len = snprintf(buffer, 128, "xpsr: 0x%08x\r\n", ctxt->xpsr);
        HAL_UART_Transmit(&huart, (uint8_t *)buffer, len, 500);
        len = snprintf(buffer, 128, "cfsr: 0x%08x\r\n", _CFSR);
        HAL_UART_Transmit(&huart, (uint8_t *)buffer, len, 500);
        len = snprintf(buffer, 128, "hfsr: 0x%08x\r\n", _HFSR);
        HAL_UART_Transmit(&huart, (uint8_t *)buffer, len, 500);
        len = snprintf(buffer, 128, "dfsr: 0x%08x\r\n", _DFSR);
        HAL_UART_Transmit(&huart, (uint8_t *)buffer, len, 500);
        len = snprintf(buffer, 128, "afsr: 0x%08x\r\n", _AFSR);
        HAL_UART_Transmit(&huart, (uint8_t *)buffer, len, 500);
        len = snprintf(buffer, 128, "mmar: 0x%08x\r\n", _MMAR);
        HAL_UART_Transmit(&huart, (uint8_t *)buffer, len, 500);
        len = snprintf(buffer, 128, "bfar: 0x%08x\r\n", _BFAR);
        HAL_UART_Transmit(&huart, (uint8_t *)buffer, len, 500);
        LED0_OFF;
        delay_dumb(100);
    }
}
