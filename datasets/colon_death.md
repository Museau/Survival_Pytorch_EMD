# COLON dataset - Chemotherapy for Stage B/C colon cancer

It is available [here](https://vincentarelbundock.github.io/Rdatasets/datasets.html).

## Description

These are data from one of the first successful trials of adjuvant chemotherapy for colon cancer. Levamisole is a low-toxicity compound previously used to treat worm infestations in animals; 5-FU is a moderately toxic (as these things go) chemotherapy agent. There are two records per person, one for recurrence and one for death.

## Columns

- id: id
- study: 1 for all patients
- rx: treatment - Obs(ervation), Lev(amisole), Lev(amisole)+5-FU
- sex: 0=female, 1=male
- age: in years
- obstruct: obstruction of colon by tumour. 0=no, 1=yes
- perfor: perforation of colon. 0=no, 1=yes
- adhere: adherence to nearby organs. 0=no, 1=yes
- nodes: number of lymph nodes with detectable cancer
- status: censoring status
- differ: differentiation of tumour (1=well, 2=moderate, 3=poor)
- extent: extent of local spread (1=submucosa, 2=muscle, 3=serosa, 4=contiguous structures)
- surg: time from surgery to registration (0=short, 1=long)
- node4: more than 4 positive lymph nodes. 0=no, 1=yes
- time: days until event or censoring
- etype: event type: 1=recurrence, 2=death

## Size

(1858, 16)

## Example

"id","study","rx","sex","age","obstruct","perfor","adhere","nodes", \
"status","differ","extent","surg","node4","time","etype"
1,1,"Lev+5FU",1,43,0,0,0,5,1,2,3,0,1,1521,2
1,1,"Lev+5FU",1,43,0,0,0,5,1,2,3,0,1,968,1
2,1,"Lev+5FU",1,63,0,0,0,1,0,2,3,0,0,3087,2
2,1,"Lev+5FU",1,63,0,0,0,1,0,2,3,0,0,3087,1

## Note

The study is originally described in Laurie (1989). The main report is found in Moertel (1990). This data set is closest to that of the final report in Moertel (1991). A version of the data with less follow-up time was used in the paper by Lin (1994).

## References

- JA Laurie, CG Moertel, TR Fleming, HS Wieand, JE Leigh, J Rubin, GW McCormack, JB Gerstner, JE Krook and J Malliard. Surgical adjuvant therapy of large-bowel carcinoma: An evaluation of levamisole and the combination of levamisole and fluorouracil: The North Central Cancer Treatment Group and the Mayo Clinic. J Clinical Oncology, 7:1447-1456, 1989.
- DY Lin. Cox regression analysis of multivariate failure time data: the marginal approach. Statistics in Medicine, 13:2233-2247, 1994.
- CG Moertel, TR Fleming, JS MacDonald, DG Haller, JA Laurie, PJ Goodman, JS Ungerleider, WA Emerson, DC Tormey, JH Glick, MH Veeder and JA Maillard. Levamisole and fluorouracil for adjuvant therapy of resected colon carcinoma. New England J of Medicine, 332:352-358, 1990.
- CG Moertel, TR Fleming, JS MacDonald, DG Haller, JA Laurie, CM Tangen, JS Ungerleider, WA Emerson, DC Tormey, JH Glick, MH Veeder and JA Maillard, Fluorouracil plus Levamisole as an effective adjuvant therapy after resection of stage II colon carcinoma: a final report. Annals of Internal Med, 122:321-326, 1991.
