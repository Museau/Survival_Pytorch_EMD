# Aids 3 - Australian AIDS Survival Data

It is available [here](https://vincentarelbundock.github.io/Rdatasets/datasets.html).

Transformation of Aids2 into Aids3 should be done beforehand using the R script Aids2_to_Aids3.R

## Description:

Data on patients diagnosed with AIDS in Australia before 1 July 1991.

## Columns

- idno: ID number.
- zid:  The factor zid indicates whether the patient is likely to have received zidovudine at all, and if so whether it might have been administered during HIV infection. Values: {0: 582, 1: 1851, 2: 1552}.
- start: # days of diagnosis.
- stop: # days of death or end of observation.
- status: 0: 2223, 1: 1762
- state: grouped state of origin: 'NSW', 'Other', 'QLD', 'VIC'. "NSW "includes ACT and "other" is WA, SA, NT and TAS.
- T.categ: reported transmission category. 'blood', 'haem', 'het', 'hs', 'hsid', 'id', 'mother', 'other'
- age: age (years) at diagnosis.
- sex: sex of patient. 'F', 'M'

## Size

(3985, 9)

## Example

"idno","zid","start","stop","status","state","T.categ","age","sex"
"1","1",0,176.9,1,"NSW","het",35,"M"
"2","1",0,67.8999999999996,1,"NSW","het",53,"M"
"3","0",0,432.9,1,"NSW","het",42,"M"
"4","0",0,77.8999999999996,1,"NSW","hsid",44,"M"

## Note

This data set has been slightly jittered as a condition of its release, to ensure patient confidentiality.

## Source

Dr P. J. Solomon and the Australian National Centre in HIV Epidemiology and Clinical Research.


## References:

Venables, W. N. and Ripley, B. D. (2002) Modern Applied Statistics with S. Fourth edition. Springer.
