SELECT * FROM [Covid '19 ]..CovidWorldData
SELECT COUNT([location]) FROM [Covid '19 ]..CovidWorldData
SELECT COUNT(continent) FROM [Covid '19 ]..CovidWorldData

-- checking again after ordering the data  
SELECT * FROM [Covid '19 ]..CovidWorldData
ORDER BY [date]

-- let's check most affected countries
SELECT [location], total_cases, total_deaths,[date]  
FROM [Covid '19 ]..CovidWorldData
ORDER BY total_deaths DESC  

SELECT [location], total_cases, total_deaths,[date]  
FROM [Covid '19 ]..CovidWorldData
ORDER BY [date] ASC  


SELECT [location], total_cases, total_deaths,[date]  
FROM [Covid '19 ]..CovidWorldData
WHERE [location] IS  NULL
ORDER BY total_deaths DESC 

SELECT [continent], [location], total_cases, total_deaths,[date]  
FROM [Covid '19 ]..CovidWorldData
WHERE [continent] IS  NULL
ORDER BY total_deaths DESC  
-- around 9.5k values are missing in the continent column

SELECT continent, location, new_cases, new_deaths, [date] 
FROM [Covid '19 ]..CovidWorldData
ORDER BY [date] ASC

SELECT location, MAX(cast(new_cases as int)) as infection_toll, SUM(cast(new_deaths as int)) as death_toll 
FROM [Covid '19 ]..CovidWorldData
GROUP BY [location]
ORDER BY infection_toll DESC 

SELECT [location], MAX(total_cases) as Highest_Infection_count, MAX(total_deaths) as Highest_death_count,population, MAX((total_cases/population)*100) AS infection_rate, MAX((total_deaths/population)*100) AS death_rate 
FROM [Covid '19 ]..CovidWorldData
GROUP BY [location],population
ORDER BY infection_rate DESC

-- let's check mortality rate
SELECT [location], total_cases, total_deaths,population,(total_deaths/population)*100 AS death_rate, [date]  
FROM [Covid '19 ]..CovidWorldData
ORDER BY death_rate DESC 

SELECT [location], total_cases, total_deaths, MAX((total_deaths/population)*100) AS max_death_rate 
FROM [Covid '19 ]..CovidWorldData

SELECT [location], total_cases, total_deaths, MIN((total_deaths/population)*100) AS min_death_rate 
FROM [Covid '19 ]..CovidWorldData

-- 

SELECT [location], date, new_cases, new_deaths, new_cases_smoothed, new_deaths_smoothed  
FROM [Covid '19 ]..CovidWorldData   
WHERE [location] like '%India%'
ORDER BY date ASC 