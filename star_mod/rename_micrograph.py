import star_tools.star_tools as st



def main(star,output, mode = 'new'):
    star = st._open_star(star, mode)
    star._rlnMicrographName = star['_rlnMicrographName'].apply(lambda x: f'f{x[0:5]}.tomostar')
    st._writeStar(star,output,mode)

if __name__ == '__main__':
    main('/struct/mahamid/rasmus/data/ribo_memb/flipped.star', '/struct/mahamid/rasmus/data/ribo_memb/flipped_renamed_f.star' )