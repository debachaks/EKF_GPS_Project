/*
 * hpm_configure.c
 *
 *  Created on: Nov 22, 2025
 *      Author: arjun
 */
#include "uart.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <metal/cpu.h>
#include <metal/hpm.h>
#include <metal/machine.h>

#include "hpm_configure.h"

#define MAKE_EVENT_MASK(evt_class, evt_id) \
    ((evt_id) | ((evt_class) & 0xFF))

void configure_hpm_events_set1()
{
	//********HPC Segment************
	struct metal_cpu *cpu = metal_cpu_get(0);
	if (!cpu) {
	   uart_puts("Failed to get CPU handle!\n");
	}

	int rc = metal_hpm_init(cpu);
	if (rc != 0) {
		uart_puts("metal_hpm_init() failed: %d\n");
	}

	for (int counter = 3; counter <= 10; ++counter) {
        metal_hpm_counter hpm = (metal_hpm_counter)counter;

        int bit = 8 + (counter - 3);     // HPM3?bit8 ... HPM20?bit25
        unsigned int event_class = 0x00; // lower 8 bits
        unsigned int event_mask  = (1UL << bit) | event_class;

        int rc = metal_hpm_set_event(cpu, hpm, event_mask);
        if (rc != 0)
        	uart_puts("Failed to set HPM\n");
        else
        	uart_puts("HPM configured\n");
    }
}

void configure_hpm_events_set2()
{
	//********HPC Segment************
	struct metal_cpu *cpu = metal_cpu_get(0);
	if (!cpu) {
	   uart_puts("Failed to get CPU handle!\n");
	}

	int rc = metal_hpm_init(cpu);
	if (rc != 0) {
		uart_puts("metal_hpm_init() failed: %d\n");
	}

	for (int counter = 3; counter <= 10; ++counter) {
        metal_hpm_counter hpm = (metal_hpm_counter)counter;

        int bit = 15 + (counter - 3);     // HPM3?bit8 ... HPM20?bit25
        unsigned int event_class = 0x00; // lower 8 bits
        unsigned int event_mask  = (1UL << bit) | event_class;

        int rc = metal_hpm_set_event(cpu, hpm, event_mask);
        if (rc != 0)
        	uart_puts("Failed to set HPM\n");
        else
        	uart_puts("HPM configured\n");
    }
}

void configure_hpm_events_set3()
{
	//********HPC Segment************
	struct metal_cpu *cpu = metal_cpu_get(0);
	if (!cpu) {
	   uart_puts("Failed to get CPU handle!\n");
	}

	int rc = metal_hpm_init(cpu);
	if (rc != 0) {
		uart_puts("metal_hpm_init() failed: %d\n");
	}

	for (int counter = 3; counter <= 4; ++counter) {
        metal_hpm_counter hpm = (metal_hpm_counter)counter;

        int bit = 24 + (counter - 3);     // HPM3?bit8 ... HPM20?bit25
        unsigned int event_class = 0x00; // lower 8 bits
        unsigned int event_mask  = (1UL << bit) | event_class;

        int rc = metal_hpm_set_event(cpu, hpm, event_mask);
        if (rc != 0)
        	uart_puts("Failed to set HPM\n");
        else
        	uart_puts("HPM configured\n");
    }
}

void configure_hpm_events_set4()
{
	//********HPC Segment************
	struct metal_cpu *cpu = metal_cpu_get(0);
	if (!cpu) {
	   uart_puts("Failed to get CPU handle!\n");
	}

	int rc = metal_hpm_init(cpu);
	if (rc != 0) {
		uart_puts("metal_hpm_init() failed: %d\n");
	}

	for (int counter = 3; counter <= 10; ++counter) {
        metal_hpm_counter hpm = (metal_hpm_counter)counter;

        int bit = 8 + (counter - 3);     // HPM3?bit8 ... HPM20?bit25
        unsigned int event_class = 0x01; // lower 8 bits
        unsigned int event_mask  = (1UL << bit) | event_class;

        int rc = metal_hpm_set_event(cpu, hpm, event_mask);
        if (rc != 0)
        	uart_puts("Failed to set HPM\n");
        else
        	uart_puts("HPM configured\n");
    }
}

void configure_hpm_events_set5()
{
	//********HPC Segment************
	struct metal_cpu *cpu = metal_cpu_get(0);
	if (!cpu) {
	   uart_puts("Failed to get CPU handle!\n");
	}

	int rc = metal_hpm_init(cpu);
	if (rc != 0) {
		uart_puts("metal_hpm_init() failed: %d\n");
	}

	for (int counter = 3; counter <= 6; ++counter) {
        metal_hpm_counter hpm = (metal_hpm_counter)counter;

        int bit = 15 + (counter - 3);     // HPM3?bit8 ... HPM20?bit25
        unsigned int event_class = 0x01; // lower 8 bits
        unsigned int event_mask  = (1UL << bit) | event_class;

        int rc = metal_hpm_set_event(cpu, hpm, event_mask);
        if (rc != 0)
        	uart_puts("Failed to set HPM\n");
        else
        	uart_puts("HPM configured\n");
    }
}

void configure_hpm_events_set6()
{
	//********HPC Segment************
	struct metal_cpu *cpu = metal_cpu_get(0);
	if (!cpu) {
	   uart_puts("Failed to get CPU handle!\n");
	}

	int rc = metal_hpm_init(cpu);
	if (rc != 0) {
		uart_puts("metal_hpm_init() failed: %d\n");
	}

	for (int counter = 3; counter <= 10; ++counter) {
        metal_hpm_counter hpm = (metal_hpm_counter)counter;

        int bit = 8 + (counter - 3);     // HPM3?bit8 ... HPM20?bit25
        unsigned int event_class = 0x02; // lower 8 bits
        unsigned int event_mask  = (1UL << bit) | event_class;

        int rc = metal_hpm_set_event(cpu, hpm, event_mask);
        if (rc != 0)
        	uart_puts("Failed to set HPM\n");
        else
        	uart_puts("HPM configured\n");
    }
}

void configure_best_hpm_events()
{
    struct metal_cpu *cpu = metal_cpu_get(0);
    if (!cpu) {
        uart_puts("Failed to get CPU handle!\n");
        return;
    }

    int rc = metal_hpm_init(cpu);
    if (rc != 0) {
        uart_puts("metal_hpm_init() failed\n");
        return;
    }

    /* Ranked event masks now mapped sequentially to HPM3?HPM10 */
    unsigned int event_masks[8] = {
    		0x0402,
    		0x0201,
    		0x8001,
    		0x4001,
    		0x1001,
    		0x2001,
    		0x0801,
    		0x8000
    };

    for (int i = 0; i < 8; i++) {
        metal_hpm_counter hpm = (metal_hpm_counter)(3 + i);
        unsigned int mask = event_masks[i];

        int rc2 = metal_hpm_set_event(cpu, hpm, mask);
        if (rc2 != 0) {
            uart_puts("Failed to set HPM\n");
        } else {
            char msg[64];
            sprintf(msg, "HPM%d configured with mask 0x%04X\n", 3 + i, mask);
            uart_puts(msg);
        }
    }
}


void configure_hpm_events_macro_set1()
{
    struct metal_cpu *cpu = metal_cpu_get(0);
    if (!cpu) {
        uart_puts("Failed to get CPU handle!\n");
        return;
    }

    int rc = metal_hpm_init(cpu);
    if (rc != 0) {
        uart_puts("metal_hpm_init() failed\n");
        return;
    }

    unsigned int event_masks[8] = {
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_0, METAL_HPM_EVENTID_8),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_0, METAL_HPM_EVENTID_9),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_0, METAL_HPM_EVENTID_10),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_0, METAL_HPM_EVENTID_11),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_0, METAL_HPM_EVENTID_12),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_0, METAL_HPM_EVENTID_13),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_0,  METAL_HPM_EVENTID_14),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_0, METAL_HPM_EVENTID_15)

    };

    for (int i = 0; i < 8; i++) {
        metal_hpm_counter hpm = (metal_hpm_counter)(3 + i);
        unsigned int mask = event_masks[i];

        int rc2 = metal_hpm_set_event(cpu, hpm, mask);
        if (rc2 != 0) {
            uart_puts("Failed to set HPM\n");
        } else {
            char msg[64];
            sprintf(msg, "HPM%d configured with mask 0x%04X\n", 3 + i, mask);
            uart_puts(msg);
        }
    }
}

void configure_hpm_events_macro_set2()
{
    struct metal_cpu *cpu = metal_cpu_get(0);
    if (!cpu) {
        uart_puts("Failed to get CPU handle!\n");
        return;
    }

    int rc = metal_hpm_init(cpu);
    if (rc != 0) {
        uart_puts("metal_hpm_init() failed\n");
        return;
    }

    unsigned int event_masks[8] = {
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_0, METAL_HPM_EVENTID_16),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_0, METAL_HPM_EVENTID_17),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_0, METAL_HPM_EVENTID_18),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_0, METAL_HPM_EVENTID_19),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_0, METAL_HPM_EVENTID_20),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_0, METAL_HPM_EVENTID_21),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_0,  METAL_HPM_EVENTID_22),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_0, METAL_HPM_EVENTID_23)

    };

    for (int i = 0; i < 8; i++) {
        metal_hpm_counter hpm = (metal_hpm_counter)(3 + i);
        unsigned int mask = event_masks[i];

        int rc2 = metal_hpm_set_event(cpu, hpm, mask);
        if (rc2 != 0) {
            uart_puts("Failed to set HPM\n");
        } else {
            char msg[64];
            sprintf(msg, "HPM%d configured with mask 0x%04X\n", 3 + i, mask);
            uart_puts(msg);
        }
    }
}

void configure_hpm_events_macro_set3()
{
    struct metal_cpu *cpu = metal_cpu_get(0);
    if (!cpu) {
        uart_puts("Failed to get CPU handle!\n");
        return;
    }

    int rc = metal_hpm_init(cpu);
    if (rc != 0) {
        uart_puts("metal_hpm_init() failed\n");
        return;
    }

    unsigned int event_masks[8] = {
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_0, METAL_HPM_EVENTID_24),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_0, METAL_HPM_EVENTID_25),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_1, METAL_HPM_EVENTID_8),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_1, METAL_HPM_EVENTID_9),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_1, METAL_HPM_EVENTID_10),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_1, METAL_HPM_EVENTID_11),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_1,  METAL_HPM_EVENTID_12),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_1, METAL_HPM_EVENTID_13)

    };

    for (int i = 0; i < 8; i++) {
        metal_hpm_counter hpm = (metal_hpm_counter)(3 + i);
        unsigned int mask = event_masks[i];

        int rc2 = metal_hpm_set_event(cpu, hpm, mask);
        if (rc2 != 0) {
            uart_puts("Failed to set HPM\n");
        } else {
            char msg[64];
            sprintf(msg, "HPM%d configured with mask 0x%04X\n", 3 + i, mask);
            uart_puts(msg);
        }
    }
}

void configure_hpm_events_macro_set4()
{
    struct metal_cpu *cpu = metal_cpu_get(0);
    if (!cpu) {
        uart_puts("Failed to get CPU handle!\n");
        return;
    }

    int rc = metal_hpm_init(cpu);
    if (rc != 0) {
        uart_puts("metal_hpm_init() failed\n");
        return;
    }

    unsigned int event_masks[8] = {
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_1, METAL_HPM_EVENTID_14),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_1, METAL_HPM_EVENTID_15),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_1, METAL_HPM_EVENTID_16),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_1, METAL_HPM_EVENTID_17),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_1, METAL_HPM_EVENTID_18),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_1, METAL_HPM_EVENTID_19),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_2,  METAL_HPM_EVENTID_8),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_2, METAL_HPM_EVENTID_9)

    };

    for (int i = 0; i < 8; i++) {
        metal_hpm_counter hpm = (metal_hpm_counter)(3 + i);
        unsigned int mask = event_masks[i];

        int rc2 = metal_hpm_set_event(cpu, hpm, mask);
        if (rc2 != 0) {
            uart_puts("Failed to set HPM\n");
        } else {
            char msg[64];
            sprintf(msg, "HPM%d configured with mask 0x%04X\n", 3 + i, mask);
            uart_puts(msg);
        }
    }
}


void configure_hpm_events_macro_set5()
{
    struct metal_cpu *cpu = metal_cpu_get(0);
    if (!cpu) {
        uart_puts("Failed to get CPU handle!\n");
        return;
    }

    int rc = metal_hpm_init(cpu);
    if (rc != 0) {
        uart_puts("metal_hpm_init() failed\n");
        return;
    }

    unsigned int event_masks[4] = {
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_2, METAL_HPM_EVENTID_10),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_2, METAL_HPM_EVENTID_11),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_2, METAL_HPM_EVENTID_12),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_2, METAL_HPM_EVENTID_13),
    };

    for (int i = 0; i < 4; i++) {
        metal_hpm_counter hpm = (metal_hpm_counter)(3 + i);
        unsigned int mask = event_masks[i];

        int rc2 = metal_hpm_set_event(cpu, hpm, mask);
        if (rc2 != 0) {
            uart_puts("Failed to set HPM\n");
        } else {
            char msg[64];
            sprintf(msg, "HPM%d configured with mask 0x%04X\n", 3 + i, mask);
            uart_puts(msg);
        }
    }
}

void configure_hpm_events_macro_PCA_selected_set()
{
    struct metal_cpu *cpu = metal_cpu_get(0);
    if (!cpu) {
        uart_puts("Failed to get CPU handle!\n");
        return;
    }

    int rc = metal_hpm_init(cpu);
    if (rc != 0) {
        uart_puts("metal_hpm_init() failed\n");
        return;
    }


    unsigned int event_masks[8] = {
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_1, METAL_HPM_EVENTID_14),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_1, METAL_HPM_EVENTID_15),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_1, METAL_HPM_EVENTID_13),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_0, METAL_HPM_EVENTID_13),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_0, METAL_HPM_EVENTID_16),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_1, METAL_HPM_EVENTID_11),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_0,  METAL_HPM_EVENTID_15),
		MAKE_EVENT_MASK(METAL_HPM_EVENTCLASS_1, METAL_HPM_EVENTID_8)

    };


    for (int i = 0; i < 8; i++) {
        metal_hpm_counter hpm = (metal_hpm_counter)(3 + i);
        unsigned int mask = event_masks[i];

        int rc2 = metal_hpm_set_event(cpu, hpm, mask);
        if (rc2 != 0) {
            uart_puts("Failed to set HPM\n");
        } else {
            char msg[64];
            sprintf(msg, "HPM%d configured with mask 0x%04X\n", 3 + i, mask);
            uart_puts(msg);

        }
    }
}
