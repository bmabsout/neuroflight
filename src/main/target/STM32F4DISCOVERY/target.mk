F405_TARGETS    += $(TARGET)
FEATURES        += VCP SDCARD

TARGET_SRC = \
            drivers/accgyro/accgyro_fake.c \
            drivers/accgyro/accgyro_mpu.c \
            drivers/accgyro/accgyro_mpu6500.c \
            drivers/accgyro/accgyro_spi_mpu6500.c
