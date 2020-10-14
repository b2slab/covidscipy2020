-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema project_covid_CREB
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `project_covid_CREB` DEFAULT CHARACTER SET utf8 ;
USE `project_covid_CREB` ;

-- -----------------------------------------------------
-- Table `project_covid_CREB`.`patients`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `project_covid_CREB`.`patients` (
  `patient_id` INT NOT NULL AUTO_INCREMENT,
  `gender` ENUM('male', 'female') NULL,
  `age` TINYINT(3) UNSIGNED NULL,
  `height` TINYINT(3) UNSIGNED NULL,
  `weight` TINYINT(3) UNSIGNED NULL,
  `lattitude` DECIMAL(8,6) NULL,
  `longitude` DECIMAL(9,6) NULL,
  `hometown` VARCHAR(30) NULL,
  `num_people` TINYINT(3) UNSIGNED NULL,
  `civil_status` ENUM('single', 'married', 'divorced', 'separated', 'in a relationship') NULL,
  `date_creation` DATE NULL,
  PRIMARY KEY (`patient_id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `project_covid_CREB`.`symptoms`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `project_covid_CREB`.`symptoms` (
  `symptom_id` INT NOT NULL AUTO_INCREMENT,
  `description_` VARCHAR(50) NULL,
  `severity` ENUM('low', 'moderate', 'high') NULL,
  `affected_organ` VARCHAR(30) NULL,
  PRIMARY KEY (`symptom_id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `project_covid_CREB`.`patients_symptoms`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `project_covid_CREB`.`patients_symptoms` (
  `patient_id` INT NOT NULL,
  `symptom_id` INT NOT NULL,
  `suffer` TINYINT NOT NULL,
  PRIMARY KEY (`patient_id`, `symptom_id`),
  INDEX `fk_symptoms_idx` (`symptom_id` ASC) VISIBLE,
  CONSTRAINT `fk_patients`
    FOREIGN KEY (`patient_id`)
    REFERENCES `project_covid_CREB`.`patients` (`patient_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_symptoms`
    FOREIGN KEY (`symptom_id`)
    REFERENCES `project_covid_CREB`.`symptoms` (`symptom_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `project_covid_CREB`.`illnesses`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `project_covid_CREB`.`illnesses` (
  `illness_id` INT NOT NULL AUTO_INCREMENT,
  `description_` VARCHAR(50) NULL,
  `severity` ENUM('low', 'moderate', 'high') NULL,
  `affected_organ` VARCHAR(30) NULL,
  PRIMARY KEY (`illness_id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `project_covid_CREB`.`patients_illnesses`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `project_covid_CREB`.`patients_illnesses` (
  `patient_id` INT NOT NULL,
  `illness_id` INT NOT NULL,
  `suffer` TINYINT NOT NULL,
  `years_suffered` TINYINT(3) UNSIGNED NULL,
  PRIMARY KEY (`patient_id`, `illness_id`),
  INDEX `fk_illnesses_idx` (`illness_id` ASC) VISIBLE,
  CONSTRAINT `fk_patients`
    FOREIGN KEY (`patient_id`)
    REFERENCES `project_covid_CREB`.`patients` (`patient_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_illnesses`
    FOREIGN KEY (`illness_id`)
    REFERENCES `project_covid_CREB`.`illnesses` (`illness_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `project_covid_CREB`.`coronavirus`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `project_covid_CREB`.`coronavirus` (
  `covid_id` INT NOT NULL AUTO_INCREMENT,
  `description_` TINYTEXT NULL,
  PRIMARY KEY (`covid_id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `project_covid_CREB`.`patients_coronavirus`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `project_covid_CREB`.`patients_coronavirus` (
  `covid_id` INT NOT NULL,
  `patient_id` INT NOT NULL,
  `suffer` TINYINT NOT NULL,
  `time_since_suffered` TINYINT(3) NULL COMMENT 'Cuánto tiempo hace que sufriste/experimentaste la situación',
  PRIMARY KEY (`covid_id`, `patient_id`),
  INDEX `fk_patients_idx` (`patient_id` ASC) VISIBLE,
  CONSTRAINT `fk_patients`
    FOREIGN KEY (`patient_id`)
    REFERENCES `project_covid_CREB`.`patients` (`patient_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_coronavirus`
    FOREIGN KEY (`covid_id`)
    REFERENCES `project_covid_CREB`.`coronavirus` (`covid_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `project_covid_CREB`.`treatment`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `project_covid_CREB`.`treatment` (
  `treatment_id` INT NOT NULL AUTO_INCREMENT,
  `patient_id` INT NOT NULL,
  `prob_corona` DECIMAL(5,2) NULL COMMENT '100.00%',
  # `audio_file_path` nvarchar(260)
  # `audio_extraction_features`
  PRIMARY KEY (`treatment_id`),
  UNIQUE INDEX `patient_id_UNIQUE` (`patient_id` ASC) VISIBLE,
  CONSTRAINT `fk_patients`
    FOREIGN KEY (`patient_id`)
    REFERENCES `project_covid_CREB`.`patients` (`patient_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
